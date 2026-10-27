"""Student demo: choose, analyse, explore."""
import hashlib
import json
from pathlib import Path
import streamlit as st
from wavesense.ingest import load_recording
from wavesense.model import predict
from wavesense.report import export_html
from wavesense.visualization import SOURCE, contributions, plate, signals


@st.cache_data(show_spinner=False)
def read(path, modified, size):
    return load_recording(path)


def recording_at(path):
    path=Path(path)
    stat=path.stat()
    return read(str(path),stat.st_mtime_ns,stat.st_size)


def main():
    st.set_page_config(page_title='WaveSense · College demo',page_icon='〰️',layout='wide')
    st.markdown('''<style>
    .stApp{background:#0b1422;color:#e4edf7}
    .block-container{max-width:1200px;padding-top:3.5rem}
    h1,h2,h3{letter-spacing:-.025em}
    [data-testid="stMetric"]{background:#14243a;padding:16px;border-radius:14px}
    </style>''',unsafe_allow_html=True)
    st.title('WaveSense')
    st.write('Can recorded sound help us spot changes in a carbon-fibre plate?')
    st.caption('Choose a recording → analyse it → explore the plate. No equipment needed.')
    try:
        artifact=json.loads(Path('outputs/model/model.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):
        st.info('The demo needs to be prepared once. See README.md for setup.')
        return
    records=artifact['partitions']
    st.subheader('1. Choose a recording')
    source=st.radio('Use an example or your own file',['Try an example','Upload a file'],horizontal=True)
    with st.form('choose_recording'):
        path,upload=None,None
        if source=='Try an example':
            lookup={r['id']:r for r in records}
            ids=list(lookup)
            default=next((i for i,r in enumerate(records) if r['split']=='test'),0)
            def label(value):
                r=lookup[value]
                condition='Healthy plate' if r['condition']=='healthy' else f'Object attached at {r["condition"]}'
                return f'{condition} · {r["temperature_C"]:.0f} °C · example {ids.index(value)+1}'
            selected=st.selectbox('Example recording',ids,index=default,format_func=label)
            path=Path(lookup[selected]['source_file'])
            st.caption('“Object attached” means a removable laboratory reference object, not a confirmed internal crack.')
        else:
            upload=st.file_uploader('Choose a recording file',type=['h5','hdf5'])
            st.caption('Original OGW #2 files only: 40 kHz excitation, 10 MHz sampling. Start with an example if you do not have one.')
        submitted=st.form_submit_button('Analyse recording',type='primary',width='stretch')
    if submitted:
        st.session_state.pop('analysis',None)
        try:
            if source=='Upload a file':
                if upload is None:
                    raise ValueError('Choose a file first, or switch to Try an example.')
                data=upload.getvalue()
                path=Path('outputs/uploads')/f'{hashlib.sha256(data).hexdigest()}.h5'
                path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():
                    path.write_bytes(data)
            with st.spinner('Reading the sound recordings…'):
                recording=recording_at(path)
                result=predict(recording,artifact)
            st.session_state.analysis=(recording,result)
        except (ValueError,OSError,KeyError) as exc:
            st.error(f'We could not analyse this recording. {exc}')
    if 'analysis' not in st.session_state:
        st.info('Press Analyse recording to see the result and explore the plate.')
        with st.expander('What is this project?'):
            st.write('Researchers sent ultrasound through a carbon-fibre plate and recorded it using 12 small devices called sensors. We test whether a computer can distinguish a healthy plate from one with a reference object attached. Temperature can change the sound too.')
        return
    recording,result=st.session_state.analysis
    meta=recording.metadata
    st.subheader('2. See the result')
    st.caption(f'Last analysed: {meta["id"]}. After choosing another example, press Analyse recording again.')
    known='Healthy plate' if meta['condition']=='healthy' else 'Unknown' if meta['condition']=='unknown' else f'Object at {meta["condition"]}'
    a,b=st.columns(2)
    a.metric('Recorded experiment',known)
    b.metric('Computer prediction',result['result'])
    st.caption('Healthy-like = resembles healthy examples. Reference-damage-like = resembles examples with an attached object.')
    if meta['condition']!='unknown':
        agrees=(meta['condition']=='healthy')==(result['result']=='Healthy-like')
        st.write('**This prediction matches the experiment label.**' if agrees else '**This prediction is wrong for this example.**')
    tn,fp=artifact['evaluations'][0]['confusion'][0]
    st.warning(f'Learning demo: the model wrongly flagged {fp} of {tn+fp} healthy recordings in its hotter test set. A prediction is not proof of damage.')
    st.subheader('3. Explore the plate')
    st.caption('The dots are sensors. The highlighted line joins the two sensors whose recording you are viewing.')
    pairs=[tuple(map(int,p)) for p in recording.channels]
    cols=st.columns([2,1])
    pair=cols[0].selectbox('Connection to explore',pairs,index=pairs.index((1,7)) if (1,7) in pairs else 0,
        format_func=lambda p:f'Sensor {p[0]} ↔ Sensor {p[1]}')
    view=cols[1].radio('View',['3D','Top'],horizontal=True)
    if st.button('Reset view'):
        st.session_state['camera_reset']=st.session_state.get('camera_reset',0)+1
    specimen=plate(pair,meta['condition'],three_d=view=='3D')
    specimen.update_layout(uirevision=f'{view}-{st.session_state.get("camera_reset",0)}')
    st.plotly_chart(specimen,width='stretch',config={'scrollZoom':False,'displaylogo':False,
        'modeBarButtonsToRemove':['toImage','zoom3d','pan3d','orbitRotation','tableRotation','resetCameraLastSave3d']})
    st.caption('Drag to rotate. Blue = sensors · green = selected connection · orange = known reference-object position. The orange mark is not a prediction.')
    references=[]
    for name,healthy in [('Healthy comparison',True),('Attached-object comparison',False)]:
        candidates=[r for r in artifact['references'] if (r['condition']=='healthy')==healthy and r['sha256']!=meta['sha256']]
        candidates.sort(key=lambda r:abs(r['temperature_C']-(meta['temperature_C'] or 20)))
        if candidates:
            try:
                reference=recording_at(candidates[0]['source_file'])
                if reference.metadata['sha256']!=candidates[0]['sha256']:
                    raise ValueError('Comparison file changed since training')
                references.append((name,reference))
            except (OSError,ValueError) as exc:
                st.caption(f'Comparison unavailable: {exc}')
    evidence=signals(recording,pair,references)
    with st.expander('See the recorded sound'):
        st.write('Time runs left to right. Curve height is the recorded response. Temperature and the attached object can both change its shape.')
        st.plotly_chart(evidence[0],width='stretch')
        st.caption('Comparisons are chosen automatically from training examples by temperature proximity. Changing the displayed connection does not change the prediction.')
    with st.expander('Details for your project report'):
        st.write(f'Model score: **{result["score"]:.3f}**. Below 0.5 means healthy-like. This is not a probability of damage.')
        st.write(f'{result["available_paths"]} connections · {result["sensors_used"]} sensors · {meta["temperature_C"] if meta["temperature_C"] is not None else "Unknown"} °C')
        st.caption(f'500 × 500 × 2 mm plate. {SOURCE}. Sensor dots are position markers, not physical sizes or depths. Connection lines are not simulated waves.')
        st.write(f'Experiment partition: {result["split"]}. Training examples were used to fit the model; they are not independent tests.')
        for warning in result['warnings']:
            st.caption(warning)
        st.plotly_chart(evidence[1],width='stretch')
        st.plotly_chart(evidence[2],width='stretch')
        st.plotly_chart(contributions(result),width='stretch')
        st.caption('Advanced comparisons: streamlit run app/research_dashboard.py')
    report={**result,'comparison_ids':[r.metadata['id'] for _,r in references],
            'displayed_pair':list(pair),'geometry_source':SOURCE}
    st.download_button('Save my report',export_html(report,[specimen,*evidence]),'wavesense-report.html','text/html',width='stretch')
    st.caption('Opens in a browser and includes your result, plots and limitations.')


if __name__=='__main__':
    main()
