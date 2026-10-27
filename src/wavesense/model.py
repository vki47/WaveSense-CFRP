"""Inspectable linear models and fixed temperature-block experiments."""
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from wavesense.ingest import load_recording
from wavesense.processing import CONFIG, features

LIMITATIONS = [
    'Prototype on a small deterministic archive subset; scores are uncalibrated, not physical damage probabilities.',
    'One specimen and removable surface reference defects; internal delamination and other structures are not validated.',
    'Condition and recording cycle are confounded. Temperature holdout does not establish independent-cycle generalization.',
    'No spatial localization is implemented. Path contributions do not identify defect positions.',
]


def partition(records):
    """Assign whole acquisition groups using fixed chamber-temperature boundaries."""
    groups = {}
    for record in records:
        t = record['temperature_C']
        if t is None:
            raise ValueError('Training requires chamber temperature metadata')
        split = 'train' if t <= 38 else 'test' if t >= 42 else 'validation'
        group = record['acquisition_group']
        if group in groups and groups[group] != split:
            raise ValueError('Temperatures disagree across frequencies in an acquisition group')
        groups[group] = split
    return np.array([groups[r['acquisition_group']] for r in records])


def metrics(y, probability):
    prediction = probability >= .5
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return dict(n=len(y), healthy_n=int(tn+fp), reference_n=int(tp+fn),
                recall=float(tp/(tp+fn)) if tp+fn else None,
                false_alarm=float(fp/(tn+fp)) if tn+fp else None,
                balanced_accuracy=float(balanced_accuracy_score(y, prediction)) if len(np.unique(y)) == 2 else None,
                macro_f1=float(f1_score(y, prediction, labels=[0, 1], average='macro', zero_division=0)),
                confusion=[[int(tn), int(fp)], [int(fn), int(tp)]])


def fit_linear(x, y, selected):
    selected = np.asarray(selected)
    flat = x[:, selected, :].reshape(len(x), -1)
    mean = flat.mean(axis=0)
    scale = np.maximum(flat.std(axis=0), 1e-6)
    estimator = LogisticRegression(C=.1, class_weight='balanced', max_iter=2000, random_state=47)
    estimator.fit((flat-mean)/scale, y)
    return dict(selected=selected.tolist(), mean=mean.tolist(), scale=scale.tolist(),
                coefficient=estimator.coef_[0].tolist(), intercept=float(estimator.intercept_[0]))


def score(model, x, unavailable=()):
    selected = np.array(model['selected'])
    z = (x[:, selected, :].reshape(len(x), -1)-model['mean']) / model['scale']
    z = z.reshape(len(x), len(selected), -1)
    z[:, np.isin(selected, list(unavailable)), :] = 0  # Explicit train-mean fallback.
    contributions = (z * np.array(model['coefficient']).reshape(len(selected), -1)).sum(axis=2)
    return expit(contributions.sum(axis=1) + model['intercept']), contributions


def train(manifest, output):
    all_records = manifest['records']
    # Validate grouping across frequencies even though this first model uses one frequency.
    partition([r for r in all_records if r['condition'] != 'unknown'])
    records = [r for r in all_records if r['condition'] != 'unknown' and r['frequency_Hz'] == 40000]
    if len(records) < 8:
        raise ValueError('Need at least eight labeled 40 kHz acquisitions; run wavesense subset')
    splits = partition(records)
    loaded = [load_recording(r['source_file']) for r in records]
    channels = loaded[0].channels
    if len(channels) != 66 or any(not np.array_equal(r.channels, channels) for r in loaded):
        raise ValueError('Training requires all 66 identical sensor pairs in every recording')
    if any(r.metadata['sha256'] != m['sha256'] for r, m in zip(loaded, records)):
        raise ValueError('Raw files changed after manifest creation; rebuild the manifest')
    x = np.stack([features(r) for r in loaded])
    y = np.array([r['condition'] != 'healthy' for r in records], dtype=int)
    tr, te = splits == 'train', splits == 'test'
    if any(np.sum(y[mask] == cls) < 2 for mask in (tr, te) for cls in (0, 1)):
        raise ValueError('Need at least two healthy and two reference acquisitions in both fixed train/test bands')
    # All rankings use training features/labels only. No tuning on the holdout.
    healthy, damage = x[tr & (y == 0)], x[tr & (y == 1)]
    variance = np.maximum(x[tr].std(axis=0), .02)
    separation = np.mean(np.abs(damage.mean(axis=0)-healthy.mean(axis=0))/variance, axis=1)
    thermal = np.mean(healthy.std(axis=0)/variance, axis=1)
    rankings = {'damage-only': np.argsort(-separation), 'thermal-only': np.argsort(thermal),
                'TASP': np.argsort(-(separation/(thermal+.1)))}
    models, evaluations = {}, []
    def add(name, selected, method, seed=None):
        model = fit_linear(x[tr], y[tr], selected)
        models[name] = model
        p, _ = score(model, x[te])
        evaluations.append(dict(model=name, method=method, seed=seed, paths=len(selected),
            sensors=len(np.unique(channels[selected])), **metrics(y[te], p)))
    add('all-path', np.arange(66), 'all-path')
    for budget in (6, 12, 24):
        for method, ranking in rankings.items():
            add(f'{method}-{budget}', ranking[:budget], method)
        for seed in range(5):
            add(f'random-{budget}-{seed}', np.random.default_rng(seed).choice(66, budget, replace=False), 'random', seed)
    baseline = models['all-path']
    probability, _ = score(baseline, x[te])
    stress = []
    for sensor in range(1, 13):
        missing = np.flatnonzero(np.any(channels == sensor, axis=1))
        p, _ = score(baseline, x[te], missing)
        available = np.delete(channels, missing, axis=0)
        stress.append(dict(kind='transducer outage', scenario=f'T{sensor}', missing_paths=len(missing),
                           paths=len(available), sensors=len(np.unique(available)), **metrics(y[te], p)))
    for fraction in (.1, .25, .5):
        for seed in range(5):
            missing = np.random.default_rng(seed).choice(66, round(66*fraction), replace=False)
            p, _ = score(baseline, x[te], missing)
            available = np.delete(channels, missing, axis=0)
            stress.append(dict(kind='path dropout', scenario=f'{fraction:.0%}; seed {seed}',
                missing_paths=len(missing), paths=len(available), sensors=len(np.unique(available)), **metrics(y[te], p)))
    predictions = [dict(id=r['id'], condition=r['condition'], temperature_C=r['temperature_C'],
                        score=float(p), actual=int(label)) for r, p, label in zip(np.array(records, dtype=object)[te], probability, y[te])]
    by_condition = []
    test_conditions = np.array([r['condition'] for r in np.array(records, dtype=object)[te]])
    for condition in sorted(set(test_conditions)-{'healthy'}):
        mask = np.isin(test_conditions, ['healthy', condition])
        by_condition.append(dict(condition=condition, **metrics(y[te][mask], probability[mask])))
    content = dict(format_version=1, frequency_Hz=40000, sampling_frequency_Hz=loaded[0].metadata['sampling_frequency_Hz'],
        config=CONFIG, channels=channels.tolist(), models=models, evaluations=evaluations, stress=stress,
        predictions=predictions, by_condition=by_condition, limitations=LIMITATIONS,
        partitions=[dict(**r, split=str(s)) for r, s in zip(records, splits)],
        training_temperature_C=[min(r['temperature_C'] for r, t in zip(records, tr) if t), max(r['temperature_C'] for r, t in zip(records, tr) if t)],
        protocol='Fixed train <=38 C, reserved validation 38–42 C, test >=42 C; threshold 0.5; C=0.1; no tuning or calibration',
        ranking_definition='D=mean standardized class-mean difference; V=mean standardized healthy SD; TASP=D/(V+0.1). No redundancy penalty.',
        references=[r for r, t in zip(records, tr) if t])
    content['software'] = dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ['model.py', 'processing.py', 'ingest.py']})
    content['version'] = 'baseline-' + hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:12]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'model.json').write_text(json.dumps(content, indent=2), encoding='utf-8')
    return content


def predict(recording, artifact, model_name='all-path', disabled_sensors=(), disabled_paths=()):
    meta = recording.metadata
    if meta['frequency_Hz'] != artifact['frequency_Hz'] or meta['sampling_frequency_Hz'] != artifact['sampling_frequency_Hz']:
        raise ValueError('This model supports 40 kHz excitation at 10 MHz sampling only; retraining is needed for other frequencies')
    model = artifact['models'][model_name]
    expected = [tuple(c) for c in artifact['channels']]
    observed = {tuple(c): f for c, f in zip(recording.channels, features(recording))}
    if any(c not in expected for c in observed):
        raise ValueError('Sensor pair direction is incompatible with the trained channel mapping')
    x = np.zeros((1, len(expected), len(CONFIG['windows_us'])))
    missing = []
    for i, pair in enumerate(expected):
        if pair not in observed or any(s in disabled_sensors for s in pair) or i in disabled_paths:
            missing.append(i)
        else:
            x[0, i] = observed[pair]
    available = [i for i in model['selected'] if i not in missing]
    if not available:
        raise ValueError('No selected measurement paths remain; a result cannot be computed')
    probability, contributions = score(model, x, missing)
    p = float(probability[0])
    warnings = list(LIMITATIONS)
    if missing:
        warnings.append(f'{len(missing)} paths unavailable: explicitly replaced with training feature means (zero contribution); score may be unreliable.')
    if .35 <= p <= .65:
        warnings.append('Weak separation: score is near the fixed 0.5 decision boundary.')
    t = meta['temperature_C']
    lo, hi = artifact['training_temperature_C']
    if t is None or not lo <= t <= hi:
        warnings.append('Temperature is unknown or outside the training range; extrapolation may be unreliable.')
    matched = next((r for r in artifact['partitions'] if r['sha256'] == meta['sha256'] or r['acquisition_group'] == meta['acquisition_group']), None)
    split = matched['split'] if matched else 'new / not in saved experiment'
    if split == 'train':
        warnings.append('This acquisition was used for training; its prediction is not a validation result.')
    return dict(input=meta, model_version=artifact['version'], model=model_name, config=artifact['config'],
        result='Reference-damage-like' if p >= .5 else 'Healthy-like', score=p,
        score_meaning='Uncalibrated reference-damage-like classifier score; threshold 0.5',
        split=split, available_paths=len(available), sensors_used=len(set(v for i in available for v in expected[i])),
        missing_paths=missing, disabled_sensors=list(disabled_sensors), disabled_paths=list(disabled_paths),
        warnings=warnings, intercept=model['intercept'], contributions=[dict(path_index=int(i),
            pair=list(expected[i]), log_odds=float(c), available=i in available)
            for i, c in zip(model['selected'], contributions[0])])
