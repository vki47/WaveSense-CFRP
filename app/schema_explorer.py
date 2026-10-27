"""Local HDF5 schema and waveform explorer; no diagnosis is performed here."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from wavesense.audit import inspect_hdf5, read_trace


st.set_page_config(page_title="WaveSense CFRP", page_icon="〰️", layout="wide")
st.title("WaveSense-CFRP")
st.caption("Local guided-wave data audit · schema exploration only · no diagnosis")

default_dir = Path("data/raw")
with st.sidebar:
    st.header("Recording")
    typed_path = st.text_input("HDF5 file path", value="")
    uploaded = st.file_uploader("Or choose an HDF5 file", type=["h5", "hdf5"])
    st.caption("Raw data stays local. Uploaded data is processed in this running session.")

path: Path | None = None
if uploaded is not None:
    target_dir = Path("outputs/uploads")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / Path(uploaded.name).name
    path.write_bytes(uploaded.getvalue())
elif typed_path.strip():
    path = Path(typed_path).expanduser()
else:
    found = sorted(p for p in default_dir.rglob("*") if p.suffix.lower() in {".h5", ".hdf5"}) if default_dir.exists() else []
    if found:
        path = st.selectbox("Local sample", found, format_func=lambda p: str(p.relative_to(Path.cwd())))

if path is None:
    st.info("Download a schema sample with `wavesense sample`, or choose a local HDF5 file above.")
    st.code("wavesense sample\nwavesense audit\nstreamlit run app/streamlit_app.py", language="powershell")
    st.stop()

if not path.is_file():
    st.error(f"File not found: {path}")
    st.stop()

try:
    schema = inspect_hdf5(path)
except (OSError, ValueError) as exc:
    st.error(f"Could not inspect this HDF5 file: {exc}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("File size", f"{schema['bytes'] / 1024**2:.2f} MiB")
c2.metric("Datasets", len(schema["datasets"]))
c3.metric("SHA-256", schema["sha256"][:12] + "…")

with st.expander("Recording metadata", expanded=True):
    st.json(schema["root_attributes"])

if not schema["datasets"]:
    st.warning("No datasets were found in this HDF5 file.")
    st.stop()

st.subheader("Waveform explorer")
numeric = []
with h5py.File(path, "r") as handle:
    for row in schema["datasets"]:
        ds = handle[row["path"]]
        if np.issubdtype(ds.dtype, np.number) and not np.issubdtype(ds.dtype, np.complexfloating) and ds.ndim >= 1 and ds.shape[0] >= 8:
            numeric.append(row)

if not numeric:
    st.info("No real numeric datasets are currently eligible for trace plotting.")
else:
    selected = st.selectbox("Numeric dataset", numeric, format_func=lambda row: f"{row['path']} · shape {row['shape']}")
    shape = selected["shape"]
    default_axis = 1 if len(shape) > 1 else 0
    sample_axis = st.number_input("Axis containing time samples", min_value=0, max_value=len(shape)-1, value=default_axis, step=1)
    other_axes = [axis for axis in range(len(shape)) if axis != sample_axis]
    indices = []
    controls = st.columns(max(1, len(other_axes)))
    for j, axis in enumerate(other_axes):
        indices.append(controls[j % len(controls)].number_input(f"Index for axis {axis} (size {shape[axis]})", min_value=0,
            max_value=max(0, shape[axis]-1), value=0, step=1, key=f"axis_{axis}"))
    try:
        trace = read_trace(path, selected["path"], int(sample_axis), tuple(int(i) for i in indices))
        fig = go.Figure(go.Scatter(y=trace, mode="lines", name="amplitude"))
        fig.update_layout(xaxis_title="Sample index (time axis not yet mapped)", yaxis_title="Recorded value", height=420, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The x-axis is a sample index until the dataset's sampling interval and axis meaning are verified.")
    except (OSError, ValueError, IndexError) as exc:
        st.error(str(exc))

with st.expander("HDF5 structure"):
    st.json({"groups": schema["groups"], "datasets": schema["datasets"]})

st.warning("A plotted waveform is a raw recording. This page does not assign temperature, sensor, condition, or damage labels, and does not produce an inspection result.")
