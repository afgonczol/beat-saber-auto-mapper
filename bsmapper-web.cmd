@echo off
rem Double-click to start the bsmapper web UI and open it in the browser.
rem Close this window to stop the server.
cd /d "%~dp0"
if not exist ".venv\Scripts\bsmapper.exe" (
    echo No .venv found next to this file. See docs\GETTING_STARTED.md.
    pause
    exit /b 1
)
rem Open the page once the server has had a few seconds to load the model.
start "" /min cmd /c "timeout /t 8 /nobreak >nul & start http://127.0.0.1:8765/"
".venv\Scripts\bsmapper.exe" serve
pause
