"""Train-only healthy temperature correction followed by logistic classification."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from wavesense.ingest import load_recording
from wavesense.model import metrics
from wavesense.model import LIMITATIONS, predict as baseline_predict
from wavesense.processing import CONFIG, features


def basis(temperatures, degree):
    t=(np.asarray(temperatures,dtype=float)-40)/20
    if not np.isfinite(t).all():
        raise ValueError('Temperature must be finite for temperature correction')
    return np.stack([t**i for i in range(degree+1)],axis=1)


def corrected(x, temperatures, model):
    flat=np.asarray(x).reshape(len(x),-1)
    if model['degree']:
        flat=flat-basis(temperatures,model['degree'])@np.asarray(model['temperature_coefficients'])
    return flat


def fit(x, y, temperatures, degree=1, C=.1):
    y=np.asarray(y)
    model=dict(degree=degree,C=C,temperature_coefficients=None)
    if degree:
        healthy=y==0
        design=basis(np.asarray(temperatures)[healthy],degree)
        if np.sum(healthy)<degree+2 or np.linalg.matrix_rank(design)<degree+1:
            raise ValueError('Not enough distinct healthy temperatures to fit this trend')
        model['temperature_coefficients']=np.linalg.lstsq(design,x[healthy].reshape(np.sum(healthy),-1),rcond=None)[0].tolist()
    flat=corrected(x,temperatures,model)
    mean=flat.mean(axis=0)
    scale=np.maximum(flat.std(axis=0),1e-6)
    clf=LogisticRegression(C=C,class_weight='balanced',max_iter=2000,random_state=47)
    clf.fit((flat-mean)/scale,y)
    model.update(mean=mean.tolist(),scale=scale.tolist(),coefficient=clf.coef_[0].tolist(),intercept=float(clf.intercept_[0]))
    return model


def scores(x, temperatures, model):
    z=(corrected(x,temperatures,model)-model['mean'])/model['scale']
    parts=(z*np.asarray(model['coefficient'])).reshape(len(x),66,4).sum(axis=2)
    return expit(parts.sum(axis=1)+model['intercept']),parts


def load_development(baseline_path):
    baseline=json.loads(Path(baseline_path).read_text(encoding='utf-8'))
    records=baseline['partitions']
    loaded=[]
    for metadata in records:
        r=load_recording(metadata['source_file'])
        if r.metadata['sha256']!=metadata['sha256']:
            raise ValueError('Development file changed; do not silently change the frozen cohort')
        if r.channels.tolist()!=baseline['channels']:
            raise ValueError('Training requires the same complete 66-channel mapping')
        loaded.append(r)
    return baseline,records,np.stack([features(r) for r in loaded])


def develop(baseline_path=Path('outputs/model/model.json'),output=Path('outputs/presentation')):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    baseline,records,x=load_development(baseline_path)
    y=np.array([r['condition']!='healthy' for r in records],dtype=int)
    t=np.array([r['temperature_C'] for r in records],dtype=float)
    groups=np.array([r['acquisition_group'] for r in records])
    conditions=np.array([r['condition'] for r in records])
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=47).split(x,conditions,groups))
    candidates=[]
    for degree in (0,1,2):
        for C in (.01,.1,1.):
            out=np.zeros(len(x))
            for tr,va in folds:
                model=fit(x[tr],y[tr],t[tr],degree,C)
                out[va]=scores(x[va],t[va],model)[0]
            row=dict(degree=degree,C=C,**metrics(y,out))
            candidates.append(row)
            print(json.dumps(row),flush=True)
    best=min(candidates,key=lambda r:(-r['balanced_accuracy'],r['false_alarm'],r['degree'],r['C']))
    model=fit(x,y,t,best['degree'],best['C'])
    artifact=dict(kind='temperature-logistic-v1',algorithm='Supervised binary classification: regularized logistic regression',
        model=model,config=CONFIG,frequency_Hz=40000,sampling_frequency_Hz=10000000,
        channels=baseline['channels'],development_records=records,references=records,
        training_temperature_C=[float(t.min()),float(t.max())],cv_candidates=candidates,selected_cv=best,
        folds=[dict(train_groups=groups[tr].tolist(),validation_groups=groups[va].tolist()) for tr,va in folds],
        baseline_version=baseline['version'],protocol_file='docs/DEMO_PROTOCOL.md',threshold=.5)
    artifact['version']='temperature-'+hashlib.sha256(json.dumps(artifact,sort_keys=True).encode()).hexdigest()[:12]
    target=output/'frozen_model.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8'))!=artifact:
            raise ValueError('Frozen model already exists with different contents; use another output directory')
    else:
        target.write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    print('Selected:',best,'Frozen model:',artifact['version'],flush=True)
    return artifact


def predict_recording(recording, artifact):
    """Input labels never enter the decision; they are used only by the UI/evaluator."""
    meta=recording.metadata
    if meta['frequency_Hz']!=artifact['frequency_Hz'] or meta['sampling_frequency_Hz']!=artifact['sampling_frequency_Hz']:
        raise ValueError('This model needs 40 kHz excitation and 10 MHz sampling')
    model=artifact['model']
    temperature=meta['temperature_C']
    if model['degree'] and temperature is None:
        raise ValueError('The selected temperature-corrected model needs chamber temperature metadata')
    expected=[tuple(c) for c in artifact['channels']]
    observed={tuple(c):f for c,f in zip(recording.channels,features(recording))}
    if any(pair not in expected for pair in observed):
        raise ValueError('Channel direction is incompatible with the trained model')
    x=np.zeros((1,66,4))
    missing=[]
    for i,pair in enumerate(expected):
        if pair in observed:
            x[0,i]=observed[pair]
        else:
            missing.append(i)
    if len(missing)==66:
        raise ValueError('No supported measurement paths remain')
    z=(corrected(x,[temperature],model)-model['mean'])/model['scale']
    z=z.reshape(66,4)
    z[missing]=0
    parts=(z*np.asarray(model['coefficient']).reshape(66,4)).sum(axis=1)
    probability=float(expit(parts.sum()+model['intercept']))
    warnings=list(LIMITATIONS)
    warnings[2]='Same-specimen, same-cycle acquisition holdout only; neighboring recordings remain correlated. Independent cycles, temperatures and specimens are not validated.'
    if missing:
        warnings.append(f'{len(missing)} missing paths use training means and zero model contribution; this model has not been validated for missing paths.')
    if .35<=probability<=.65:
        warnings.append('Weak evidence: this score is close to the 0.5 decision boundary.')
    if temperature is None or not artifact['training_temperature_C'][0]<=temperature<=artifact['training_temperature_C'][1]:
        warnings.append('Temperature is unknown or outside the development range; prediction may be unreliable.')
    if np.max(np.abs(z))>6:
        warnings.append('Some signal features are far outside typical training values; the score may be unreliable.')
    known=next((r for r in artifact['development_records'] if r['sha256']==meta['sha256'] or r['acquisition_group']==meta['acquisition_group']),None)
    split='train' if known else 'test' if any(r['sha256']==meta['sha256'] for r in artifact.get('test_records',[])) else 'new / not in saved experiment'
    if split=='train':
        warnings.append('This recording was used to fit the model. Its prediction is not an independent accuracy test.')
    available=[p for i,p in enumerate(expected) if i not in missing]
    return dict(input=meta,model_version=artifact['version'],model='Logistic regression',
        config={**artifact['config'],'temperature_trend_degree':model['degree'],'regularization_C':model['C']},
        result='Reference-damage-like' if probability>=.5 else 'Healthy-like',score=probability,
        score_meaning='Uncalibrated reference-damage-like classifier score; threshold 0.5',
        split=split,available_paths=len(available),sensors_used=len(set(v for p in available for v in p)),
        missing_paths=missing,warnings=warnings,intercept=model['intercept'],
        contributions=[dict(path_index=i,pair=list(pair),log_odds=float(parts[i]),available=i not in missing) for i,pair in enumerate(expected)])


def evaluate_test(root=Path('data/presentation_test'),output=Path('outputs/presentation'),baseline_path=Path('outputs/model/model.json')):
    """Evaluate only after freeze; results do not alter model selection or coefficients."""
    root,output=Path(root),Path(output)
    frozen_path=output/'frozen_model.json'
    frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    frozen_hash=hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    baseline=json.loads(Path(baseline_path).read_text(encoding='utf-8'))
    if baseline['version']!=frozen['baseline_version']:
        raise ValueError('Original baseline changed after the second experiment was frozen')
    records,predictions,original_scores,new_scores,labels=[],[],[],[],[]
    dev_groups={r['acquisition_group'] for r in frozen['development_records']}
    dev_hashes={r['sha256'] for r in frozen['development_records']}
    seen=set()
    for condition in ['healthy','D04','D12','D16','D24']:
        plan=json.loads((root/f'plan_{condition}.json').read_text(encoding='utf-8'))
        provenance=json.loads((root/f'provenance_{condition}.json').read_text(encoding='utf-8'))
        if plan['frozen_model_sha256']!=frozen_hash or provenance['frozen_model_sha256']!=frozen_hash:
            raise ValueError('Model checksum does not match the pre-download freeze')
        if len(plan['members'])!=8 or plan['members']!=[r['local_path'] for r in provenance['members']]:
            raise ValueError('Incomplete or changed test cohort')
        for member in provenance['members']:
            source=(root/member['local_path']).resolve()
            if not source.is_relative_to(root.resolve()):
                raise ValueError('Test source lies outside the test directory')
            r=load_recording(source)
            meta=r.metadata
            if meta['sha256']!=member['sha256'] or meta['condition']!=condition:
                raise ValueError('Test provenance or condition mismatch')
            if meta['acquisition_group'] in dev_groups or meta['sha256'] in dev_hashes:
                raise ValueError('Development/test overlap detected')
            if meta['acquisition_group'] in seen:
                raise ValueError('Duplicate test acquisition')
            seen.add(meta['acquisition_group'])
            new=predict_recording(r,frozen)
            old=baseline_predict(r,baseline)
            records.append(meta)
            labels.append(int(condition!='healthy'))
            new_scores.append(new['score'])
            original_scores.append(old['score'])
            predictions.append(dict(id=meta['id'],condition=condition,temperature_C=meta['temperature_C'],
                actual=labels[-1],score=new['score'],prediction=new['result'],original_score=old['score']))
    summary=dict(new_model=metrics(np.array(labels),np.array(new_scores)),
                 original_model=metrics(np.array(labels),np.array(original_scores)))
    summary['protocol']='40 new acquisition timestamps; same specimen and cycles; threshold/model frozen before download; not unseen-temperature or independent-cycle validation'
    frozen.update(test_records=records,test_predictions=predictions,test_results=summary,
                  partitions=[dict(**{k:v for k,v in r.items() if k!='split'},split='train') for r in frozen['development_records']]+[dict(**r,split='test') for r in records],
                  frozen_model_sha256=frozen_hash)
    # Preserve first evaluation; reruns must reproduce it exactly.
    target=output/'model.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=frozen:
        raise ValueError('Existing final evaluation differs; refusing to overwrite its evidence')
    target.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return frozen


if __name__=='__main__':
    develop()
