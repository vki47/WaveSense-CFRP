# Local vertical slice — 25 October 2026

## Data and provenance

The initial workspace contained one healthy 100 kHz sample and staged audit/explorer code. That sample and the original explorer are preserved. The new subset contains 60 real 40 kHz files: twelve timestamps per condition (healthy, D04, D12, D16, D24). Selection uses equally spaced indices in each sorted archive, fixed before inspecting scores. This is a manageable demonstration subset, not a representative statistical sample. Raw files, checksums, archive URLs and license metadata remain in ignored `data/`; models and reports remain in ignored `outputs/`.

Primary documentation: Moll et al., *Temperature affected guided wave propagation in a composite plate complementing the Open Guided Waves Platform*, DOI [10.1038/s41597-019-0208-1](https://doi.org/10.1038/s41597-019-0208-1), especially Tables 1–3. [Accessible source PDF](https://d-nb.info/1197303693/34). Archive collection: [10.6084/m9.figshare.c.4488089.v1](https://doi.org/10.6084/m9.figshare.c.4488089.v1).

The actual healthy and D04 HDF5 files were inspected before defining the reader; all downloaded files subsequently passed it. `pitchcatch/catch` is 66 rows × 13,108 samples; channel pairs are zero-based in the file and displayed as T1–T12. Sampling frequency is read from metadata (10 MHz), as is carrier frequency. CTC temperature is used for splitting; both surface temperatures are retained. Native timestamp is retained without guessing its epoch. Timestamp-folder names group acquisitions across frequencies. Unknown cycles remain unknown; a folder timestamp is not claimed to identify an independent experimental cycle. The first/second channel endpoints are provisionally recorded as transmitter/receiver; pair visualization does not claim independently verified direction.

Each recording and path manifest retains source path, SHA-256, condition and label source, chamber and surface temperatures, session, unknown cycle, frequency, sample rate and acquisition group. Duplicate file contents are excluded from training manifests. Corrupt files appear in manifest errors. Published folder names provide labels; arbitrary uploads do not.

## Processing and prediction

Per-trace mean removal, then log10 RMS in four fixed windows: 150–350, 350–600, 600–950 and 950–1250 microseconds. The first window starts after the 125 microsecond, five-cycle 40 kHz excitation plus a fixed margin. This is a coarse engineering baseline, not a verified mode-isolation window. No filtering, waveform alignment or temperature compensation is applied. UI waveform decimation is display-only.

One training example is an entire acquisition at 40 kHz, never an individual path. Standardization is fit on training features only. Logistic regression uses fixed C=0.1, balanced class weights and threshold 0.5. The JSON artifact contains coefficients, scaler values, selected paths, software/source fingerprints, partitions, references, metrics and configuration. No pickle loading is needed. Scores are uncalibrated. Weak scores still return a class with a warning. Unknown/out-of-range temperatures warn rather than suppress a computable result.

Missing paths and simulated transducer outages are explicitly listed. Their features use training means, contributing zero standardized evidence. This fallback is stress-tested but not trained for missingness; it can fail badly. Zero remaining selected paths is unprocessable. Incompatible frequency, sample rate, channel direction, non-finite data, missing required fields or insufficient waveform duration return errors rather than invented predictions.

## Fixed experiments

Training: chamber temperature <=38 °C. Reserved validation: 38–42 °C (unused). Test: >=42 °C. These boundaries and model settings were fixed before the first run. Group assignments are checked across frequencies; only 40 kHz enters this baseline. All rankings and scaling use training only. Comparison examples are selected from training pools, not holdout labels. Copied training files are identified by checksum for an explicit in-sample warning.

The baseline is fit first. At budgets 6, 12 and 24, separate models compare damage-only, thermal-only, exploratory TASP and five random subsets. All-path (66) is the full-resource comparator, not an equal-budget competitor to a six-path model. D is mean standardized healthy/reference class-mean separation; V is standardized healthy variability across the sampled training temperatures; TASP = D/(V+0.1). This initial score has no redundancy penalty or hyperparameter search. Healthy variability is a thermal-stability proxy, not an isolated causal temperature effect.

Experiments separately remove 10%, 25% and 50% of paths (five seeds each) and all paths incident to each of twelve transducers. Saved metrics include reference recall, healthy false alarms, balanced accuracy, macro F1, confusion counts, number of paths and unique sensors. Per-reference-condition comparisons reuse the same healthy test pool and are not independent tests. Random-subset error bars are standard deviations, not confidence intervals.

**Observed all-path result:** 25 held-out acquisitions, five healthy and twenty reference. All were predicted reference-damage-like: confusion matrix [[0,5],[0,20]], recall 1.0, false-alarm rate 1.0, balanced accuracy 0.5, macro F1 0.4444. This failure is exposed in the UI. No threshold or features were revised to improve the observed holdout. Larger future experiments need a fresh untouched test set; repeatedly consulting this holdout is exploratory research.

## Interface and exports

The dashboard provides dataset-ID selection and local HDF5 upload/path entry, score warnings, measured waveform overlays, FFT magnitude, unaligned residuals, exact linear path contributions, 2D/rotatable 3D geometry, experimental comparisons and outages. Reference examples can be chosen from training recordings; proximity ordering uses temperature, not the unknown input label.

Geometry uses documented 500×500×2 mm dimensions, sensor coordinates from Table 1, reference-position centers from Table 2. Sensor markers are projected onto the surface, with no claimed embedment depth. Orange marks are known experimental positions inferred from the published folder label, never predictions. There is no damage-region heatmap or localization. A straight path line is a sensor connection, not a simulated wave field.

HTML exports embed Plotly and plots for offline use. JSON records input/checksum, configuration, model version, inference, known condition, missing measurements, contributions and warnings. Local paths appear in exports. Experiment JSON includes full partitions and reproducibility metadata. PDF export is not implemented.

## Still pending

- More representative sampling, verified cycle IDs and independent-cycle/specimen assessment.
- Temperature compensation and meaningful reliability improvements; the current baseline demonstrably fails on healthy high-temperature cases.
- Calibration with adequate independent calibration data, uncertainty intervals and leave-one-position-out evaluation.
- Redundancy-aware TASP and properly nested tuning before any superiority claim.
- Other excitation frequencies and file formats; geometry-based localization only if justified by an implemented, validated method.

## Run and verify

```powershell
Set-Location C:\Projects\WaveSense-CFRP
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m wavesense.cli subset --count 12
.\.venv\Scripts\python.exe -m wavesense.cli train
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

The app runs at http://127.0.0.1:8501 and is restricted to localhost. No Docker, cloud service, database or GPU is needed. Model/data files already exist locally for this delivery; the download and train steps are only needed to recreate them. Tests use synthetic fixtures solely for software checks plus a smoke test on the real prepared dataset (skipped when local model artifacts are absent). The leakage test changes held-out signals and verifies that every fitted model remains identical.
