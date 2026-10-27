"""Fixed signal features; no label-dependent windows or test-derived parameters."""
import numpy as np

CONFIG = dict(version='window-log-rms-v1', windows_us=[[150, 350], [350, 600], [600, 950], [950, 1250]],
              dc_removal='per waveform mean', filter='none', alignment='none',
              note='150 us excludes the 125 us excitation at 40 kHz plus a fixed margin; no mode isolation')


def features(recording):
    fs = recording.metadata['sampling_frequency_Hz']
    signal = recording.signals - recording.signals.mean(axis=1, keepdims=True)
    values = []
    for start, end in CONFIG['windows_us']:
        a, b = int(round(start * fs / 1e6)), int(round(end * fs / 1e6))
        if b > signal.shape[1] or b <= a:
            raise ValueError('Recording is too short for the fixed 150–1250 microsecond analysis window')
        rms = np.sqrt(np.mean(signal[:, a:b] ** 2, axis=1))
        values.append(np.log10(np.maximum(rms, 1e-12)))
    result = np.stack(values, axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Signal magnitude is outside supported numeric limits')
    return result
