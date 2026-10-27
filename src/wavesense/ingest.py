"""Strict reader for the published OGW #2 pitch-catch schema."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd

CONDITIONS = {'OGW_CFRP_Temperature_udam': 'healthy', **{
    f'OGW_CFRP_Temperature_dam_{c}': c for c in ('D04', 'D12', 'D16', 'D24')}}
SOURCE = 'https://doi.org/10.1038/s41597-019-0208-1'


@dataclass
class Recording:
    metadata: dict
    channels: np.ndarray
    signals: np.ndarray


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def load_recording(path):
    path = Path(path)
    if path.suffix.lower() not in {'.h5', '.hdf5'}:
        raise ValueError('Supported input: native OGW #2 .h5 or .hdf5 pitch-catch recording')
    try:
        with h5py.File(path, 'r') as f:
            def scalar(key):
                a = np.asarray(f[key][()])
                if a.size != 1 or not np.isfinite(a).all():
                    raise ValueError(f'{key} must contain one finite value')
                return float(a.ravel()[0])
            ds = f['pitchcatch/catch']
            if ds.ndim != 2 or not 1 <= ds.shape[0] <= 66 or not 8 <= ds.shape[1] <= 200_000:
                raise ValueError('Expected 1–66 waveform rows and 8–200,000 samples per row')
            if ds.dtype.kind not in 'fiu':
                raise ValueError('Waveforms must be real numeric values')
            signals = np.asarray(ds[:], dtype=float)
            channels = np.asarray(f['command/pitchcatch/channels'][:])
            if channels.shape != (len(signals), 2) or channels.dtype.kind not in 'iu':
                raise ValueError('Channel table must contain two integer endpoints per waveform')
            if np.any(channels < 0) or np.any(channels > 11) or np.any(channels[:, 0] == channels[:, 1]):
                raise ValueError('Expected distinct OGW transducer indices 0–11')
            if len(np.unique(np.sort(channels, axis=1), axis=0)) != len(channels):
                raise ValueError('Duplicate sensor pair in recording')
            if not np.isfinite(signals).all():
                raise ValueError('Waveform contains NaN or infinity')
            fs = scalar('command/pitchcatch/sampling_frequency')
            frequency = scalar('command/pitchcatch/signal_frequency')
            if not 0 < frequency < fs / 2:
                raise ValueError('Sampling and excitation frequencies are inconsistent')
            temperature = scalar('CTC/Temperature') if 'CTC/Temperature' in f else None
            surface = np.asarray(f['Temperature/values'][:], dtype=float) if 'Temperature/values' in f else None
            if surface is not None and (surface.shape != (2,) or not np.isfinite(surface).all()):
                raise ValueError('Surface temperatures must contain two finite values')
            timestamp = scalar('timestamp') if 'timestamp' in f else None
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError(f'Unreadable or unsupported OGW recording: {exc}') from exc
    condition = next((CONDITIONS[p] for p in path.parts if p in CONDITIONS), 'unknown')
    session = path.parent.name if re.fullmatch(r'\d{8}T\d{6}', path.parent.name) else None
    digest = checksum(path)
    # Timestamp groups are label-independent: all frequencies from one acquisition stay together.
    group = session or (f'native-{timestamp:.10f}' if timestamp is not None else digest)
    record_id = f'{condition}/{session or digest[:16]}/{frequency/1000:g}kHz'
    metadata = dict(id=record_id, source_file=str(path.resolve()), sha256=digest,
        condition=condition, label_source='published folder name' if condition != 'unknown' else 'unknown',
        temperature_C=temperature, surface_temperature_C=surface.tolist() if surface is not None else None,
        session=session, cycle='unknown', acquisition_group=group, native_timestamp=timestamp,
        frequency_Hz=frequency, sampling_frequency_Hz=fs, samples=signals.shape[1], paths=len(channels))
    order = np.lexsort((channels[:, 1], channels[:, 0]))
    return Recording(metadata, channels[order].astype(int) + 1, signals[order])


def build_manifest(root, output):
    records, rows, errors, seen = [], [], [], set()
    for path in sorted(Path(root).rglob('*')):
        if path.suffix.lower() not in {'.h5', '.hdf5'}:
            continue
        try:
            recording = load_recording(path)
            meta = recording.metadata
            if meta['sha256'] in seen:
                continue
            seen.add(meta['sha256'])
            records.append(meta)
            rows.extend({**meta, 'transmitter': int(a), 'receiver': int(b),
                         'channel_order_note': 'first/second native endpoints; assumed Tx/Rx order'}
                        for a, b in recording.channels)
        except ValueError as exc:
            errors.append(dict(source_file=str(path), error=str(exc)))
    result = dict(records=records, errors=errors, schema_source=SOURCE)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(output / 'paths.csv', index=False)
    return result
