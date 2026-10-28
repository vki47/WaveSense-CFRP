# WaveSense-CFRP

WaveSense-CFRP is a local workbench for exploring public ultrasonic guided-wave recordings from CFRP specimens. It now includes a working baseline classifier, a dark inspection dashboard, measured signal comparisons, documented 2D/3D geometry, grouped experiments and portable HTML/JSON reports. The original generic HDF5 explorer is preserved in `app/schema_explorer.py`.

**Scientific status:** this is an end-to-end software prototype, not a reliable damage detector. On the first 25 hotter held-out acquisitions, the all-path baseline classified every recording as reference-damage-like: 100% reference recall, 100% healthy false alarms and 50% balanced accuracy. No model or threshold was tuned after seeing these results. See `docs/IMPLEMENTATION.md` for exact scope, assumptions and pending research.

## Run the prepared application

The default app is a short college-demo workflow: choose an example (or upload a supported file), press **Analyse recording**, and explore the plate. Graphs and technical details are optional expanders. Use **Top** for a flat map and **Reset view** to restore the 3D camera.

**Presentation update:** the default app now prefers the prepared model in `outputs/presentation/model.json`, falling back to the original model if unavailable. This is supervised logistic regression trained across the development temperature range. It correctly classified 40/40 additional acquisitions (8 healthy and 32 reference-object cases). These come from the same specimen and cycles; this does not validate real internal damage detection. The original failed experiment above remains part of the project history.

Double-click `Launch_WaveSense.cmd` to start. Read [the presentation and viva guide](docs/PRESENTATION_GUIDE.md) and [verified results and demo examples](docs/DEMO_RESULTS.md). Two standalone HTML reports are prepared under `outputs/presentation/` for an offline backup.

The improved seven-slide presentation is saved in [docs/presentation](docs/presentation). Download or clone the repository and open `docs/presentation/index.html` in a browser. Use arrow keys to advance and F11 for full screen. These slides are image exports of the editable presentation; [speaker notes](docs/presentation/SPEAKER_NOTES.md) are included separately.

After preparing the original 60-file baseline below, reproduce the second experiment with `python -m wavesense.cli prepare-demo`, then `python scripts/prepare_demo_reports.py`. This downloads 40 additional files, freezes model selection before reading them, and preserves the first saved evaluation. See [the frozen protocol](docs/DEMO_PROTOCOL.md). Raw data and model outputs remain local and are not stored in Git.

The advanced dashboard, model comparisons and outage controls are preserved separately:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/research_dashboard.py --server.port 8502
```

From `C:\Projects\WaveSense-CFRP` in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open http://127.0.0.1:8501. Select a dataset ID or a local native OGW HDF5 file. The first model supports **40 kHz excitation and 10 MHz sampling**, including partial channel sets when at least one selected path remains. Other formats/frequencies produce an explicit error. The original 100 kHz sample remains available through the generic explorer.

To reproduce the local preparation (60 files, approximately 795 MiB uncompressed):

```powershell
.\.venv\Scripts\python.exe -m wavesense.cli subset --count 12
.\.venv\Scripts\python.exe -m wavesense.cli train
.\.venv\Scripts\python.exe -m pytest -q
```

`subset` retrieves twelve equally spaced archive timestamps per condition at 40 kHz, checks archive CRCs and records SHA-256 provenance. It skips already downloaded valid files. `train` reads raw files without changing them and rebuilds ignored outputs under `outputs/model/`: per-path manifest, saved model and experiment results. Add your own native data under `data/raw/`, keeping the published folder names if known labels are required for training. An uploaded file has an unknown label unless its original labeled folder path is supplied; filenames alone do not establish its true condition.

## Set up on Windows

Use Python 3.11 or later. From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks environment activation, run the commands without activation by replacing `python` with `.\.venv\Scripts\python.exe`.

## First data audit

Download one HDF5 member from the healthy OGW Dataset #2 archive. This bounded schema sample is for file-format inspection only; it is not a representative or statistically valid model dataset.

```powershell
wavesense sample
wavesense audit
streamlit run app/schema_explorer.py
```

The sample downloader checks that the remote server honours byte-range requests, enforces a per-file size limit, verifies ZIP integrity and writes a provenance JSON file with source metadata and a SHA-256 digest. Raw data is kept under `data/raw/`, which Git intentionally ignores.

To retrieve a sample from a reference-damage condition:

```powershell
wavesense sample --condition D04
```

Available conditions are `healthy`, `D04`, `D12`, `D16` and `D24`. A sample from one condition is useful for schema exploration only; do not use it to evaluate a classifier.

The generic audit writes `outputs/audit/schema_audit.json` and `outputs/audit/dataset_inventory.csv`. The preserved schema explorer displays arbitrary numeric traces against sample index. The new OGW-specific dashboard maps waveform rows and sample timing using the published schema and file sampling frequency.

## Git workflow

This repository uses `main` as its default branch. Start each piece of work by checking where you are and what changed:

```powershell
git status --short --branch
git diff
```

For a future reviewable checkpoint, the following commands illustrate staging and committing. **They were not run for this delivery.** Existing staged work is preserved, and new work is left unstaged/untracked:

```powershell
git add README.md pyproject.toml .gitignore src app
git diff --cached
git commit -m "Build local OGW data audit workbench"
```

For a small solo project, committing directly on `main` is fine if you prefer to keep the workflow simple. To practise feature branches, create one before future work, then merge it back into `main` after reviewing it. A commit stays local until you explicitly push it with `git push`.

## Current limits and next milestone

This is a software-only prototype with an uncalibrated trained model. OGW Dataset #2 includes a single principal specimen and experimental reference damage; results must not be presented as validation for arbitrary structures or internal delamination. A high classifier score can still be wrong, as the held-out healthy false alarms demonstrate.

Next research steps are temperature compensation, larger and independently grouped evaluation, cycle mapping, calibration, and leave-one-reference-position-out testing. No damage-region heatmap, defect depth, shape or inferred location is currently supported.
