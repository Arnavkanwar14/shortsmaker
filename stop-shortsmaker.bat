@echo off
rem ---- shortsmaker: stop the web UI ----
cd /d "%~dp0"

if exist "shortsmaker.pid" (
    set /p SMPID=<shortsmaker.pid
    taskkill /PID %SMPID% /T /F >nul 2>&1
    del shortsmaker.pid
)

rem fallback: kill whatever is still listening on port 8000
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo shortsmaker stopped.
