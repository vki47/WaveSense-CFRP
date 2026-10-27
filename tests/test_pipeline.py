"""Synthetic fixtures verify software behavior only, never scientific performance."""
import itertools
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from wavesense.ingest import Recording, build_manifest, load_recording
from wavesense.model import fit_linear, partition, predict, score, train
from wavesense.processing import CONFIG, features
from wavesense.report import export_html
from wavesense.visualization import plate


def fixture_file(path, temperature=20, factor=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = np.array(list(itertools.combinations(range(12), 2)), dtype='uint8')
    time = np.arange(13108)/1e7
    signal = factor*np.sin(2*np.pi*40000*time)[None,:]*np.linspace(.1,1,66)[:,None]
    with h5py.File(path,'w') as f:
        f['pitchcatch/catch']=signal
        f['command/pitchcatch/channels']=channels
        f['command/pitchcatch/sampling_frequency']=[1e7]
        f['command/pitchcatch/signal_frequency']=[40000.]
        f['CTC/Temperature']=[temperature]
    return path


def test_reader_schema_units_and_manifest(tmp_path):
    path=fixture_file(tmp_path/'OGW_CFRP_Temperature_udam'/'20260101T120000'/'pc_f40kHz.h5')
    r=load_recording(path)
    assert r.metadata['condition']=='healthy'
    assert r.metadata['cycle']=='unknown'
    assert r.channels[0].tolist()==[1,2]
    assert features(r).shape==(66,4)
    manifest=build_manifest(tmp_path,tmp_path/'output')
    assert len(manifest['records'])==1 and not manifest['errors']
    assert (tmp_path/'output'/'paths.csv').read_text().count('\n')==67


@pytest.mark.parametrize('kind',['nan','channels','short','missing'])
def test_invalid_inputs(tmp_path,kind):
    path=fixture_file(tmp_path/'test.h5')
    with h5py.File(path,'a') as f:
        if kind=='nan': f['pitchcatch/catch'][0,0]=np.nan
        elif kind=='channels': f['command/pitchcatch/channels'][0]=[0,0]
        elif kind=='short':
            del f['pitchcatch/catch']; f['pitchcatch/catch']=np.ones((66,100))
        else: del f['command/pitchcatch/sampling_frequency']
    with pytest.raises(ValueError): features(load_recording(path))


def test_grouping_across_frequencies():
    rows=[dict(acquisition_group='a',temperature_C=20),dict(acquisition_group='a',temperature_C=21),dict(acquisition_group='b',temperature_C=45)]
    assert partition(rows).tolist()==['train','train','test']
    rows[1]['temperature_C']=45
    with pytest.raises(ValueError,match='disagree'): partition(rows)


def test_missing_paths_and_contribution_identity(tmp_path):
    recording=load_recording(fixture_file(tmp_path/'test.h5'))
    x=np.stack([features(recording)+i*.2 for i in range(4)])
    model=fit_linear(x,np.array([0,0,1,1]),np.arange(66))
    artifact=dict(models={'all-path':model},frequency_Hz=40000,sampling_frequency_Hz=1e7,
        channels=recording.channels.tolist(),training_temperature_C=[20,38],partitions=[],version='test',config=CONFIG)
    result=predict(recording,artifact,disabled_sensors=[1])
    assert result['available_paths']==55 and result['sensors_used']==11
    assert sum(not r['available'] for r in result['contributions'])==11
    logit=result['intercept']+sum(r['log_odds'] for r in result['contributions'])
    assert result['score']==pytest.approx(1/(1+np.exp(-logit)))
    with pytest.raises(ValueError,match='No selected'): predict(recording,artifact,disabled_sensors=list(range(1,13)))
    recording.metadata['frequency_Hz']=100000
    with pytest.raises(ValueError,match='40 kHz'): predict(recording,artifact)


def test_holdout_cannot_change_fitted_models(tmp_path):
    for condition in ['udam','dam_D04']:
        for i,t in enumerate([20,30,45,50]):
            fixture_file(tmp_path/f'OGW_CFRP_Temperature_{condition}'/f'2026010{i+1}T{12 if condition=="udam" else 13}0000'/'pc_f40kHz.h5',t,1+i*.1+(condition!='udam'))
    first=train(build_manifest(tmp_path,tmp_path/'out'),tmp_path/'out')
    for r in first['partitions']:
        if r['split']=='test':
            with h5py.File(r['source_file'],'a') as f: f['pitchcatch/catch'][:]*=8
    second=train(build_manifest(tmp_path,tmp_path/'out'),tmp_path/'out')
    assert first['models']==second['models']
    assert {r['acquisition_group'] for r in first['partitions'] if r['split']=='train'}.isdisjoint({r['acquisition_group'] for r in first['partitions'] if r['split']=='test'})


def test_report_escapes_metadata_and_embeds_plot():
    text=export_html(dict(result='<script>unsafe</script>',score=.5),[plate((1,7))])
    assert '<script>unsafe</script>' not in text
    assert '&lt;script&gt;' in text and 'plotly.js' in text
