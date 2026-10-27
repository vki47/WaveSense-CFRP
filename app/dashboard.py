"""Local inspection dashboard; run through streamlit_app.py."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from wavesense.ingest import load_recording
from wavesense.model import predict
from wavesense.report import export_html
from wavesense.visualization import SOURCE, contributions, plate, signals, style


@st.cache_data(show_spinner=False)
def read(path, mtime, size):
    return load_recording(path)


def cached_recording(path):
    path = Path(path)
    stat = path.stat()
    return read(str(path), stat.st_mtime_ns, stat.st_size)


def main():
    st.set_page_config(page_title='WaveSense · CFRP', page_icon='〰️', layout='wide')
    st.markdown('''<style>
    .stApp {background:radial-gradient(ellipse at top right,#15334a 0%,#0a1220 55%);color:#dce7f5}
    [data-testid="stSidebar"] {background:#0d1828;border-right:1px solid #25374b}
    .block-container {padding-top:4rem;max-width:1600px}
    h1,h2,h3 {letter-spacing:-.03em} .eyebrow {color:#67e8f9;letter-spacing:.22em;font-size:.75rem}
    [data-testid="stMetric"] {background:#101f32;border:1px solid #294057;border-radius:12px;padding:18px}
    .stTabs [data-baseweb="tab-list"] {gap:24px;border-bottom:1px solid #294057}
    </style>''', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">GUIDED-WAVE INSPECTION / LOCAL RESEARCH WORKBENCH</div>', unsafe_allow_html=True)
    st.title('WaveSense–CFRP')
    st.caption('Explore recorded ultrasound measurements from a carbon-fibre plate — one step at a time.')
    artifact_path = Path('outputs/model/model.json')
    if not artifact_path.exists():
        st.info('Prepare the local prototype data and train the model to begin.')
        st.code('.\\.venv\\Scripts\\python.exe -m wavesense.cli subset\n.\\.venv\\Scripts\\python.exe -m wavesense.cli train')
        st.caption('The subset uses 12 timestamps per condition at 40 kHz. Full archives are not required.')
        return
    try:
        artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        st.error(f'Cannot read model: {exc}. Run wavesense train again.')
        return
    records = artifact['partitions']
    with st.sidebar:
        st.markdown('### 1 · Choose a recording')
        st.caption('New here? Keep the example selected, then read Start here. Changes update the page automatically.')
        research = st.toggle('Show research controls', value=False, help='Enable alternative models and simulated sensor failures. Leave off for a first visit: all 66 paths are used.')
        mode = st.radio('Where should the recording come from?', ['Dataset ID', 'Local file'],
                        format_func=lambda v: 'Included example recordings' if v == 'Dataset ID' else 'A file on my computer')
        uploaded = None
        if mode == 'Dataset ID':
            ids = [r['id'] for r in records]
            default = next((i for i,r in enumerate(records) if r['split'] == 'test'), 0)
            lookup = {r['id']:r for r in records}
            selected_id = st.selectbox('Choose an example', ids, index=default,
                format_func=lambda v: f'{"Healthy experiment" if lookup[v]["condition"] == "healthy" else "Reference object at " + lookup[v]["condition"]} · {lookup[v]["temperature_C"]:.1f} °C · {lookup[v]["session"]}',
                help='These descriptions are known experiment labels, not predictions. The final timestamp distinguishes recordings.')
            path = Path(records[ids.index(selected_id)]['source_file'])
        else:
            uploaded = st.file_uploader('Choose an OGW recording', type=['h5','hdf5'])
            typed = st.text_input('Or paste an absolute local file path')
            path = Path(typed).expanduser() if typed else None
            if uploaded:
                data = uploaded.getvalue()
                digest = hashlib.sha256(data).hexdigest()
                path = Path('outputs/uploads') / f'{digest}.h5'
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_bytes(data)
        st.caption('Use the included examples first. File upload accepts only the original OGW #2 HDF5 format: 40 kHz sound excitation, sampled at 10 MHz. A phone audio recording or CSV will not work.')
        st.divider()
        model_name, disabled_sensors, disabled_paths = 'all-path', [], []
        if research:
            st.markdown('### Research controls')
            names = [n for n in artifact['models'] if not n.startswith('random')]
            model_name = st.selectbox('Model / path budget', names, help='All-path uses 66 measured connections. A suffix such as -6 means the model uses six connections, not six sensors.')
            with st.expander('What do the model names mean?'):
                st.markdown('**All-path:** use every connection. **Damage-only:** select connections that separated the two classes in training. **Thermal-only:** select connections that varied less among healthy training recordings. **TASP:** combine separation with stability. The number is the connection budget. None is established as a reliable detector.')
            disabled_sensors = st.multiselect('Simulate unavailable transducers', list(range(1,13)), format_func=lambda s:f'T{s}',help='A transducer is a physical device on the plate. Removing one disables every measured connection involving it.')
            disabled_paths = st.multiselect('Simulate unavailable paths', list(range(66)),
                format_func=lambda i:f'T{artifact["channels"][i][0]}–T{artifact["channels"][i][1]}',help='Remove just one connection at a time. This simulation does not change your raw files.')
            st.caption('Missing measurements use training averages as a fallback. This can make the result unreliable. With no selected measurements left, prediction is impossible.')
        else:
            st.caption('Beginner settings: all 66 connections; no simulated failures. Turn on research controls when you want to compare methods.')
        st.divider()
        st.caption(f'Model {artifact["version"]}')
    if path is None:
        st.info('Choose a file or enter its local path.')
        return
    try:
        recording = cached_recording(path)
        result = predict(recording, artifact, model_name, disabled_sensors, disabled_paths)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Cannot process recording: {exc}')
        st.info('For other HDF5 schemas or frequencies, use the original explorer: streamlit run app/schema_explorer.py')
        return
    meta = recording.metadata
    a,b,c,d = st.columns([2,1,1,1])
    a.metric('What the model predicts', result['result'],help='Healthy-like means the model places the recording on its healthy side. Reference-damage-like means it places it on its attached-reference-object side. Neither is proof of physical damage.')
    b.metric('Model score · 0 to 1', f'{result["score"]:.3f}',help='Below 0.5: healthy-like. At least 0.5: reference-damage-like. A score of 0.998 does NOT mean a 99.8% chance of damage.')
    c.metric('Connections / devices used', f'{result["available_paths"]} / {result["sensors_used"]}',help='A path is one measured connection between two transducers. Twelve devices provide 66 distinct pairs in these recordings.')
    d.metric('Temperature during recording', f'{meta["temperature_C"]:.1f} °C' if meta['temperature_C'] is not None else 'Unknown',help='The experimental chamber temperature, not the temperature of your computer. Temperature alone can change a waveform.')
    st.caption(f'Known dataset condition: **{meta["condition"]}** (source: {meta["label_source"]}) · Experiment partition: **{result["split"]}**')
    st.warning('Uncalibrated prototype score, not a probability of physical damage. Single specimen; removable surface reference defects. No inferred defect location.')
    baseline = artifact['evaluations'][0]
    tn, fp = baseline['confusion'][0]
    st.error(f'Current baseline limitation: it falsely flagged {fp} of {tn + fp} healthy recordings in the hotter test set. Use this app to learn and investigate; its predictions cannot establish whether a structure is safe.')
    for warning in result['warnings'][4:]:
        st.warning(warning)
    tabs = st.tabs(['Start here', 'Explore signals', 'Explore the plate', 'Check model performance', 'Save a report'])
    with tabs[0]:
        st.subheader('What am I looking at?')
        st.write('CFRP means carbon-fibre-reinforced polymer: carbon fibres held together by a plastic resin. Researchers attached devices called transducers to a plate. These devices send or receive ultrasound — sound above human hearing. A guided wave travels along the plate, and each recording stores how the received signal changes over time.')
        st.markdown('**The workflow:** choose a recording → inspect the measured signals → read the model’s guess → check how often it was wrong → save your observations.')
        st.markdown('**What “reference damage” means here:** researchers attached a removable aluminium disc to the surface. D04, D12, D16 and D24 name its experimental positions. These recordings do not demonstrate real internal cracks or delamination.')
        st.subheader('Understand this selected recording')
        known = 'No reference object: a healthy experiment' if meta['condition']=='healthy' else f'Reference object at {meta["condition"]}' if meta['condition']!='unknown' else 'Unknown: this file does not establish the experimental condition'
        st.write(f'**Known experimental condition:** {known}.')
        st.write(f'**Model prediction:** {result["result"]}, with a score of {result["score"]:.3f}. The score crosses the reference-like decision boundary at 0.5; it is not a calibrated probability.')
        if meta['condition'] != 'unknown':
            agrees = (meta['condition']=='healthy') == (result['result']=='Healthy-like')
            st.info('The prediction agrees with this recording’s known label. One agreement does not establish reliability.' if agrees else 'The prediction disagrees with this recording’s known label. This is a visible model error, not evidence that the experimental label should be changed.')
        st.markdown('**Try this first:**\n1. Open **Explore the plate** to see where the devices sit.\n2. Open **Explore signals** and keep the default T1–T7 connection. Compare the three curves.\n3. Choose a healthy example in the sidebar and see whether the model agrees with its known label.\n4. Open **Check model performance** to see why an impressive score can still be misleading.\n5. Use **Save a report → HTML** for a readable record of your work.')
        with st.expander('Dictionary: the terms used throughout this app',expanded=True):
            st.markdown('''| Term | Plain-language meaning |
|---|---|
| Recording / acquisition | One captured experiment, with measurements from multiple connections. |
| T1–T12 / transducer | The numbered devices that send or receive sound. |
| Path / pair | A measured connection such as T1–T7; not a visible crack or a simulated ray. |
| Waveform / signal | A curve showing the recorded response over time. |
| µs / kHz / MHz | Microseconds (millionths of a second); thousands / millions of cycles per second. Sampling at 10 MHz means 10 million samples per second. |
| Model / inference | A learned numerical rule / its prediction for a recording. |
| Training | Examples used to fit that rule. Success on them is not a fair test. |
| Validation | Examples reserved for development decisions; this prototype leaves them unused. |
| Test / holdout | Examples kept out of fitting, used to measure errors. Here they are hotter recordings. |
| Reference / baseline | A comparison example / a starting method. In this app, “all-path baseline” means the starting classifier. |
| Uncalibrated score | A model number whose value has not been shown to match real-world error probabilities. |
| Provenance / checksum | Where a file came from / a fingerprint used to identify its contents. |''')
    with tabs[1]:
        st.subheader('2 · Compare the measured responses')
        st.write('Choose one connection to look at. This changes the plots and highlighted plate connection, not which measurements the model uses. The blue curve is your selected recording; green is a healthy comparison; orange is a reference-object comparison.')
        left,right = st.columns([1,1])
        pairs = [tuple(p) for p in recording.channels]
        pair = left.selectbox('Measurement pair', pairs, format_func=lambda p:f'T{p[0]}–T{p[1]}', index=min(5,len(pairs)-1))
        reference_records = artifact['references']
        chosen = []
        for label, healthy in [('Healthy training example',True),('Reference-damage training example',False)]:
            options = [r for r in reference_records if (r['condition']=='healthy') == healthy and r['sha256'] != meta['sha256']]
            options.sort(key=lambda r:abs(r['temperature_C']-(meta['temperature_C'] or 20)))
            if options:
                choice = right.selectbox(label,options,format_func=lambda r:f'{r["condition"]} · {r["temperature_C"]:.1f} °C · {r["session"]}')
                try:
                    reference = cached_recording(choice['source_file'])
                    if reference.metadata['sha256'] != choice['sha256']:
                        raise ValueError('Reference checksum changed since training')
                    chosen.append((label,reference))
                except (OSError,ValueError) as exc:
                    st.warning(f'Comparison unavailable: {exc}')
        st.caption('Comparison examples come only from training data, ordered by temperature proximity. They do not change the classifier score. Residuals are unaligned and can reflect temperature.')
        evidence = signals(recording,pair,chosen)
        st.plotly_chart(evidence[0], width='stretch')
        st.caption('Read this graph left to right: time passes along the bottom; the curve height is recorded signal amplitude. A different shape can result from temperature, the reference object, or other measurement differences. A large peak alone does not prove damage.')
        cols = st.columns(2)
        for col,fig in zip(cols,evidence[1:]):
            col.plotly_chart(fig,width='stretch')
        st.caption('Frequency evidence (FFT) shows which frequencies appear strongly in the signal. The residual subtracts a comparison waveform from the selected waveform. Larger residuals mean greater difference, not necessarily more damage; these traces have not been time-aligned or temperature-corrected.')
        contribution_plot = contributions(result)
        st.plotly_chart(contribution_plot,width='stretch')
        st.caption('Orange bars push the model toward reference-damage-like; blue bars push it toward healthy-like. The chart shows the 12 largest contributions in magnitude (or fewer for a small path budget). These bars explain the model, not the physical location or cause of damage.')
        st.caption(f'Exact linear-model decomposition: intercept {result["intercept"]:.3f} + all path contributions = decision log-odds. Contributions are model explanations, not causal damage localization.')
    with tabs[2]:
        st.subheader('3 · Locate the devices on the plate')
        st.write('Blue dots are the documented transducer positions. The green line highlights the connection selected in Explore signals. An orange dot marks a known reference-object position from the experiment label — the model did not discover it. Drag the 3D view to rotate; use the camera reset button to return. Switch off 3D for a flat map.')
        view3d = st.toggle('Rotatable 3D view',value=True)
        specimen = plate(pair,meta['condition'],view3d)
        st.plotly_chart(specimen,width='stretch')
        st.caption(f'{SOURCE}. 500 × 500 × 2 mm plate. Sensor markers are projected onto the surface; embedment depth is not inferred. Path lines connect sensor endpoints, not simulated wave trajectories.')
        st.info('Orange marker, when present, is the published experimental reference position from the folder label. The model detects a damage-like response without locating it.')
    with tabs[3]:
        st.subheader('4 · Ask how often the model was wrong')
        st.write('These are saved experiment results, not just the selected recording. The first metrics and confusion matrix always describe the all-path baseline, even if you choose another model in research controls.')
        st.markdown('**Recall:** of the reference-object cases, how many were flagged? Higher is better. **False alarms:** of the healthy cases, how many were wrongly flagged? Lower is better. **Balanced accuracy:** averages success on healthy and reference cases, giving both classes equal weight.')
        st.subheader('Fixed temperature holdout')
        st.write(artifact['protocol'])
        counts = pd.Series([r['split'] for r in records]).value_counts()
        st.caption(' · '.join(f'{k}: {v} acquisitions' for k,v in counts.items())+' · All paths/frequencies within an acquisition remain grouped.')
        st.warning('Small, correlated sample. Reference location and recording cycle are confounded. The reserved validation band is unused; no calibrated probabilities or independent-cycle claims are available.')
        row = artifact['evaluations'][0]
        cols=st.columns(3)
        for col,key,title in zip(cols,['recall','false_alarm','balanced_accuracy'],['Reference recall','Healthy false alarms','Balanced accuracy']):
            col.metric(title,f'{row[key]:.1%}' if row[key] is not None else 'Unavailable')
        cm = go.Figure(go.Heatmap(z=row['confusion'],x=['Predicted healthy','Predicted reference'],y=['Known healthy','Known reference'],text=row['confusion'],texttemplate='%{text}',colorscale='Blues',showscale=False))
        st.plotly_chart(style(cm,'All-path holdout confusion matrix',300),width='stretch')
        st.caption('Read each row as the known experimental condition, and each column as the prediction. The diagonal cells are correct answers; the other cells are errors. Always look at the counts as well as percentages.')
        evaluation=pd.DataFrame(artifact['evaluations'])
        comparison=go.Figure()
        for method,frame in evaluation.groupby('method'):
            grouped=frame.groupby('paths')['balanced_accuracy'].agg(['mean','std']).fillna(0).reset_index()
            comparison.add_trace(go.Scatter(x=grouped.paths,y=grouped['mean'],error_y=dict(array=grouped['std']),mode='lines+markers',name=method))
        comparison.update_xaxes(title='Measurement path budget')
        comparison.update_yaxes(title='Held-out balanced accuracy',range=[0,1.05])
        st.plotly_chart(style(comparison,'Equal-budget comparisons · random: mean ± SD over 5 subsets'),width='stretch')
        st.caption('A path budget is how many connections a method gets to use. Compare methods at the same x-axis value for a fair resource comparison. Random means randomly chosen subsets; its bars show variability across five selections, not guaranteed uncertainty bounds.')
        st.caption(artifact['ranking_definition']+' All rankings and scalers fit on training only. Error bars describe subset variability, not confidence intervals.')
        st.dataframe(evaluation.drop(columns=['confusion']),hide_index=True,width='stretch')
        st.subheader('Path loss and transducer outages')
        st.caption('Path dropout removes individual connections. A transducer outage removes every connection involving one device. These are software simulations; no original recording is altered. In the tables, n is the number of test recordings, and macro F1 averages a precision/recall score across the two classes.')
        st.dataframe(pd.DataFrame(artifact['stress']).drop(columns=['confusion']),hide_index=True,width='stretch')
        st.subheader('Known reference conditions against the same healthy holdout')
        st.dataframe(pd.DataFrame(artifact['by_condition']).drop(columns=['confusion']),hide_index=True,width='stretch')
        st.download_button('Download experiment results',json.dumps(artifact,indent=2),'wavesense-experiment.json','application/json')
    with tabs[4]:
        st.subheader('Portable inspection record')
        st.write('Choose HTML for a report you can open in a browser, with interactive plots. Choose JSON for structured data you can inspect with code. Both save the current prediction, settings and warnings; neither is an engineering safety certificate.')
        report_result = {**result, 'comparison_ids':[r.metadata['id'] for _,r in chosen],
                         'displayed_pair':list(map(int,pair)), 'geometry_source':SOURCE}
        st.download_button('Download HTML report with interactive plots',export_html(report_result,[*evidence,contribution_plot,specimen]),'wavesense-report.html','text/html')
        st.download_button('Download JSON record',json.dumps(report_result,indent=2),'wavesense-report.json','application/json')
        with st.expander('Technical record: input details, settings and file fingerprint'):
            st.json(report_result)
        st.markdown('[Dataset documentation and geometry source](https://doi.org/10.1038/s41597-019-0208-1)')
        st.caption('Raw files are opened read-only. Downloads include local source paths; review them before sharing.')


if __name__ == '__main__':
    main()
