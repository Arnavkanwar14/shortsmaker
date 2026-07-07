@echo off
rem ---- shortsmaker: start the web UI and open the browser ----
cd /d "%~dp0"

rem first run: create the environment automatically
if not exist ".venv\Scripts\python.exe" (
    echo First run - creating Python environment, this takes a few minutes...
    python -m venv .venv || (echo Python 3.10+ is required & pause & exit /b 1)
    ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
    ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
)

rem already running? just open the browser
powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/runs' -TimeoutSec 2 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }"
if %errorlevel%==0 (
    echo shortsmaker is already running.
    start "" http://127.0.0.1:8000
    exit /b 0
)

echo Starting shortsmaker...
powershell -NoProfile -Command "$p = Start-Process -FilePath '.venv\Scripts\python.exe' -ArgumentList '-m','shortsmaker','web' -WindowStyle Hidden -PassThru -RedirectStandardOutput 'web.log' -RedirectStandardError 'web.err.log'; Set-Content -Path 'shortsmaker.pid' -Value $p.Id -Encoding ascii"

rem wait (max ~30s) for the server to come up, then open the browser
powershell -NoProfile -Command "for ($i = 0; $i -lt 30; $i++) { try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/runs' -TimeoutSec 2 -UseBasicParsing | Out-Null; exit 0 } catch { Start-Sleep 1 } }; exit 1"
if %errorlevel%==0 (
    echo shortsmaker is running at http://127.0.0.1:8000
    start "" http://127.0.0.1:8000
) else (
    echo Server did not start - check web.err.log for details.
    pause
)
