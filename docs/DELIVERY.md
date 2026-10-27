# Delivery checks and file changes

Dataset loaded: 60 additional real 40 kHz recordings (12 per condition), plus the original healthy 100 kHz sample. Training partitions: 30 train, 5 reserved validation, 25 test. Raw subset size: 833,938,560 bytes.

Changed existing files:
- README.md: current setup, run commands and scientific limits.
- pyproject.toml: minimum Streamlit version for the dashboard API.
- app/streamlit_app.py: dashboard entry point.
- src/wavesense/cli.py: subset and training commands.

Added files:
- .streamlit/config.toml: dark theme, localhost binding and disabled telemetry.
- app/dashboard.py: inspection, specimen, experiment and export views.
- app/schema_explorer.py: preserved original staged explorer (content equality checked).
- src/wavesense/subset.py: bounded provenance-recorded download.
- src/wavesense/ingest.py: strict loader and per-path manifest.
- src/wavesense/processing.py: fixed waveform features.
- src/wavesense/model.py: baseline, ranking comparisons and failure stress tests.
- src/wavesense/visualization.py: evidence plots and documented geometry.
- src/wavesense/report.py: portable HTML report.
- tests/test_pipeline.py and tests/test_app.py: core and real-data UI checks.
- docs/IMPLEMENTATION.md and docs/DELIVERY.md: methodology and handoff.

Checks actually run:
- `python -m wavesense.cli train`: succeeded, zero manifest errors; real-data models and experiments saved.
- `python -m pytest -q`: 10 passed, no skips in this prepared workspace.
- `git diff --check`: succeeded, no whitespace errors (Git printed normal Windows LF/CRLF notices).
- `git check-ignore`: confirmed data, generated model and report are ignored.
- Compared preserved schema explorer against staged original: equal after newline normalization.
- Browser: loaded real predictions and visually inspected the documented 3D specimen.
- Generated offline HTML and JSON example reports under outputs/reports/.

All-path test result: 20/20 reference cases detected, but 5/5 healthy cases falsely flagged. Balanced accuracy 50%; macro F1 0.4444. This is a functioning software slice with a scientifically unsuccessful baseline, not a validated detector.

Start the app from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Open http://127.0.0.1:8501. It was left running for this delivery. Stop the server with Ctrl+C in its terminal when no longer needed.

Git status at handoff:

```
A  .gitignore
MM README.md
AM app/streamlit_app.py
AM pyproject.toml
A  src/wavesense/__init__.py
A  src/wavesense/audit.py
AM src/wavesense/cli.py
A  src/wavesense/download.py
?? .streamlit/
?? app/dashboard.py
?? app/schema_explorer.py
?? docs/
?? src/wavesense/ingest.py
?? src/wavesense/model.py
?? src/wavesense/processing.py
?? src/wavesense/report.py
?? src/wavesense/subset.py
?? src/wavesense/visualization.py
?? tests/
```

The left status column describes previously staged work; the right column describes additional working-file changes. `??` means a new file/folder has not been staged. Existing staged changes were not unstaged or replaced. No commit, push, reset or branch change was performed.
