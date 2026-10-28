"""Checks for train-only correction and frozen-model inference."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from wavesense.ingest import Recording,load_recording
from wavesense.temperature import basis,corrected,fit,predict_recording,scores


def test_healthy_trend_fit_and_serialized_prediction():
    rng=np.random.default_rng(47)
    t=np.tile(np.array([20.,30.,40.,50.,60.]),2)
    y=np.repeat([0,1],5)
    slope=rng.uniform(.1,.3,(66,4))
    x=2+((t-40)/20)[:,None,None]*slope+y[:,None,None]*.3
    model=fit(x,y,t,degree=1,C=.1)
    assert np.asarray(model['temperature_coefficients'])[1].reshape(66,4)==pytest.approx(slope)
    assert corrected(x[:5],t[:5],model)==pytest.approx(np.zeros((5,264)),abs=1e-12)
    p,parts=scores(x,t,json.loads(json.dumps(model)))
    assert (p[:5]<.5).all() and (p[5:]>=.5).all()
    assert p==pytest.approx(1/(1+np.exp(-(parts.sum(axis=1)+model['intercept']))))
    with pytest.raises(ValueError,match='finite'):
        basis([np.nan],1)


@pytest.mark.skipif(not Path('outputs/presentation/frozen_model.json').exists(),reason='No frozen demo model')
def test_prediction_never_uses_known_condition():
    artifact=json.loads(Path('outputs/presentation/frozen_model.json').read_text())
    r=load_recording(artifact['development_records'][0]['source_file'])
    p=predict_recording(r,artifact)
    other=copy.deepcopy(r)
    other.metadata['condition']='healthy' if r.metadata['condition']!='healthy' else 'D24'
    assert predict_recording(other,artifact)['score']==p['score']
    partial=Recording(copy.deepcopy(r.metadata),r.channels[11:],r.signals[11:])
    out=predict_recording(partial,artifact)
    assert out['available_paths']==55 and len(out['missing_paths'])==11
    assert all(row['log_odds']==0 for row in out['contributions'] if not row['available'])
    r.metadata['frequency_Hz']=100000
    with pytest.raises(ValueError,match='40 kHz'):
        predict_recording(r,artifact)


@pytest.mark.skipif(not Path('outputs/presentation/model.json').exists(),reason='Fresh holdout not yet evaluated')
def test_final_test_is_disjoint_and_model_frozen():
    frozen=json.loads(Path('outputs/presentation/frozen_model.json').read_text())
    final=json.loads(Path('outputs/presentation/model.json').read_text())
    assert final['model']==frozen['model'] and final['version']==frozen['version']
    for field in ['acquisition_group','sha256']:
        assert {r[field] for r in final['development_records']}.isdisjoint({r[field] for r in final['test_records']})
    assert len(final['test_records'])==40
    assert len({r['acquisition_group'] for r in final['test_records']})==40
    for r,p in zip(final['test_records'],final['test_predictions']):
        result=predict_recording(load_recording(r['source_file']),final)
        assert result['score']==pytest.approx(p['score'])
        assert result['split']=='test'
