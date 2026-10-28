@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing. Follow the setup steps in README.md.
  pause
  exit /b 1
)
echo Opening WaveSense at http://127.0.0.1:8501
echo Keep this window open during your demonstration. Press Ctrl+C to stop.
".venv\Scripts\python.exe" -m streamlit run app/streamlit_app.py
pause
