"""Create offline reports and a results sheet from the saved evaluation."""
import json
from pathlib import Path
from wavesense.ingest import load_recording
from wavesense.temperature import predict_recording
from wavesense.visualization import plate, signals
from wavesense.report import export_html

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/presentation'
a = json.loads((out / 'model.json').read_text())
rows = ['# Verified presentation results', '',
        'Model: supervised logistic regression, C=0.1; 264 log-RMS features; fixed threshold 0.5.', '',
        'Development: 60 recordings. Fresh acquisition test: 40 recordings. The model was frozen before test download.', '',
        '**40/40 correct: 8/8 healthy, 32/32 reference-object cases; zero healthy false alarms.**', '',
        'The original model scored 37/40 on this same cohort, with 3/8 healthy false alarms. Training temperature coverage changed; this is not evidence for a new temperature-correction algorithm.', '',
        '**Scope:** same specimen and recording cycles, with different timestamps. Not independent-cycle or unseen-temperature validation. The reference object is a surface disc, not real internal delamination.', '',
        '## Demo examples and offline reports', '']
for condition, name in [('healthy', 'healthy'), ('D04', 'reference')]:
    record = next(r for r in a['test_records'] if r['condition'] == condition)
    recording = load_recording(record['source_file'])
    result = predict_recording(recording, a)
    figures = [plate((1, 7), condition)] + signals(recording, (1, 7))[:2]
    (out / f'{name}_report.html').write_text(export_html(result, figures), encoding='utf-8')
    number = next(i+1 for i,r in enumerate(a['partitions']) if r['id']==record['id'])
    rows += [f'- Example {number}: {condition}, {record["temperature_C"]:.0f} °C, `{record["id"]}` → **{result["result"]}**. Offline report: `outputs/presentation/{name}_report.html`.']
rows += ['', '## All test predictions', '', '| Recording | Condition | °C | Prediction |', '|---|---|---:|---|']
for p in a['test_predictions']:
    rows.append(f'| {p["id"]} | {p["condition"]} | {p["temperature_C"]:.1f} | {p["prediction"]} |')
rows += ['', f'Model version: `{a["version"]}`.', f'Frozen model SHA-256: `{a["frozen_model_sha256"]}`.']
(root / 'docs/DEMO_RESULTS.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')
print('Saved results sheet and two offline HTML reports.')
