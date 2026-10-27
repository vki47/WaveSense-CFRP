"""Measured signal views and documented geometry, without inferred localization."""
import numpy as np
import plotly.graph_objects as go

SENSORS = {i+1: (450-80*(i % 6), 470 if i < 6 else 30) for i in range(12)}
DAMAGE = {'D04': (65, 400), 'D12': (195, 330), 'D16': (335, 260), 'D24': (450, 190)}
SOURCE = 'Moll et al. 2019, Tables 1–3, DOI 10.1038/s41597-019-0208-1'
COLORS = ['#67e8f9', '#a3e635', '#fb923c']


def style(fig, title='', height=400):
    fig.update_layout(template='plotly_dark', title=title, height=height, paper_bgcolor='#101b2b',
                      plot_bgcolor='#101b2b', font=dict(family='Segoe UI', color='#dce7f5'),
                      margin=dict(l=30, r=25, t=65, b=95), legend=dict(orientation='h', y=-.25))
    return fig


def signals(recording, pair, references=()):
    waveform, spectrum, residual = go.Figure(), go.Figure(), go.Figure()
    selected = None
    for k, (label, rec) in enumerate([('Selected', recording), *references]):
        matches = np.flatnonzero(np.all(rec.channels == pair, axis=1))
        if not len(matches):
            continue
        y = rec.signals[matches[0]]
        fs = rec.metadata['sampling_frequency_Hz']
        t = np.arange(len(y))/fs*1e6
        # Display decimation only; model always uses all samples.
        waveform.add_trace(go.Scatter(x=t[::3], y=y[::3], name=label, line_color=COLORS[k % 3]))
        f = np.fft.rfftfreq(len(y), 1/fs)/1000
        amplitude = np.abs(np.fft.rfft((y-y.mean())*np.hanning(len(y)))) / len(y)
        mask = f <= 300
        spectrum.add_trace(go.Scatter(x=f[mask], y=amplitude[mask], name=label, line_color=COLORS[k % 3]))
        if k == 0:
            selected = y
        elif len(y) == len(selected) and fs == recording.metadata['sampling_frequency_Hz']:
            residual.add_trace(go.Scatter(x=t[::3], y=(selected-y)[::3], name=f'Selected minus {label}', line_color=COLORS[k % 3]))
    waveform.update_xaxes(title='Time from record start (µs)')
    waveform.update_yaxes(title='Recorded amplitude (native units)')
    spectrum.update_xaxes(title='Frequency (kHz)')
    spectrum.update_yaxes(title='Hann-windowed FFT magnitude')
    residual.update_xaxes(title='Time from record start (µs)')
    residual.update_yaxes(title='Amplitude difference (unaligned)')
    return [style(waveform, 'Measured response'), style(spectrum, 'Frequency evidence'), style(residual, 'Unaligned reference residual')]


def plate(pair, condition='unknown', three_d=True):
    fig = go.Figure()
    if three_d:
        # Published 500 x 500 x 2 mm plate; surface markers have no claimed embedment depth.
        fig.add_trace(go.Mesh3d(x=[0,500,500,0,0,500,500,0], y=[0,0,500,500,0,0,500,500],
            z=[-2,-2,-2,-2,0,0,0,0], i=[0,0,4,4,0,0,1,1,2,2,3,3],
            j=[1,2,5,6,1,5,2,6,3,7,0,4], k=[2,3,6,7,5,4,6,5,7,6,4,7],
            color='#2b4964', opacity=1, name='CFRP plate', hoverinfo='skip',
            lighting=dict(ambient=.8,diffuse=.8,specular=.25,roughness=.7),showlegend=False))
        def add(x,y,name,mode,color,text=None):
            fig.add_trace(go.Scatter3d(x=x,y=y,z=[0]*len(x),mode=mode,name=name,text=text,
                marker=dict(size=8,color=color,line=dict(color='#091321',width=2)),
                line=dict(color=color,width=7),textposition='top center',
                textfont=dict(size=15,color='#f0f7ff'),
                hovertemplate=name+'<br>x: %{x} mm · y: %{y} mm<extra></extra>'))
        fig.add_trace(go.Scatter3d(x=[0,500,500,0,0],y=[0,0,500,500,0],z=[0]*5,
            mode='lines',line=dict(color='#6e90ad',width=3),hoverinfo='skip',showlegend=False))
        fig.update_layout(scene=dict(xaxis=dict(visible=False,range=[-30,530]),
            yaxis=dict(visible=False,range=[-30,530]),zaxis=dict(visible=False,range=[-2,0]),
            aspectmode='manual',aspectratio=dict(x=1,y=1,z=2/560),dragmode='orbit',
            camera=dict(eye=dict(x=.85,y=-1.05,z=1.25),projection=dict(type='orthographic'))))
    else:
        fig.add_shape(type='rect',x0=0,y0=0,x1=500,y1=500,line_color='#45617d',fillcolor='#172b40')
        def add(x,y,name,mode,color,text=None):
            fig.add_trace(go.Scatter(x=x,y=y,mode=mode,name=name,text=text,
                marker=dict(size=10,color=color),line=dict(color=color,width=3),textposition='top center'))
        fig.update_xaxes(title='x (mm)', range=[-30,530])
        fig.update_yaxes(title='y (mm)', range=[-30,530], scaleanchor='x')
    add([v[0] for v in SENSORS.values()], [v[1] for v in SENSORS.values()], 'Verified sensors', 'markers+text', '#67e8f9', [f'T{i}' for i in SENSORS])
    add([SENSORS[int(i)][0] for i in pair], [SENSORS[int(i)][1] for i in pair], 'Selected connection', 'lines+markers', '#a3e635')
    if condition in DAMAGE:
        x,y = DAMAGE[condition]
        add([x],[y], f'{condition}: known experimental position', 'markers+text', '#fb923c', [condition])
    style(fig, '', 580)
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10),showlegend=False,paper_bgcolor='#102034',
                      annotations=[dict(text='500 × 500 mm · 2 mm thick',xref='paper',yref='paper',
                                        x=.02,y=.02,showarrow=False,font=dict(color='#a3b9cd',size=12))])
    return fig


def contributions(result):
    ordered = sorted(result['contributions'], key=lambda c: abs(c['log_odds']))[-12:]
    fig = go.Figure(go.Bar(x=[r['log_odds'] for r in ordered],
        y=[f'T{r["pair"][0]}–T{r["pair"][1]}' for r in ordered], orientation='h',
        marker_color=['#fb923c' if r['log_odds'] > 0 else '#67e8f9' for r in ordered]))
    fig.update_xaxes(title='Contribution to log-odds: positive → reference-damage-like')
    return style(fig, 'Measurements contributing to this decision')
