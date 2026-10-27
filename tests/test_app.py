"""Smoke check the real prepared local dataset, when available."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.skipif(not Path('outputs/model/model.json').exists(),reason='Prepare local dataset first')
def test_dashboard_real_data():
    sys.path.insert(0,str(Path('app').resolve()))
    app=AppTest.from_file(Path('app/research_dashboard.py').resolve(),default_timeout=45).run()
    assert not app.exception
    assert len(app.metric)>=4
    assert app.tabs[0].label == 'Start here'
    assert not any(w.label=='Model / path budget' for w in app.selectbox)
    next(w for w in app.toggle if w.label=='Show research controls').set_value(True).run()
    next(w for w in app.selectbox if w.label=='Model / path budget').select('TASP-6').run()
    assert not app.exception
    app.multiselect[0].select(1).run()
    assert not app.exception


@pytest.mark.skipif(not Path('outputs/model/model.json').exists(),reason='Prepare local dataset first')
def test_simple_demo_flow():
    sys.path.insert(0,str(Path('app').resolve()))
    app=AppTest.from_file(Path('app/streamlit_app.py').resolve(),default_timeout=45).run()
    assert not app.exception
    assert len(app.selectbox)==1 and not app.tabs and not app.metric
    next(w for w in app.button if w.label=='Analyse recording').click().run()
    assert not app.exception and len(app.metric)==2
    original=app.session_state['analysis'][1]['input']['sha256']
    next(w for w in app.radio if w.label=='View').set_value('Top').run()
    assert not app.exception
    assert app.session_state['analysis'][1]['input']['sha256']==original
    next(w for w in app.radio if w.label=='Use an example or your own file').set_value('Upload a file').run()
    next(w for w in app.button if w.label=='Analyse recording').click().run()
    assert not app.exception and app.error
    assert not app.metric  # Failed new input must not leave an old prediction visible.
