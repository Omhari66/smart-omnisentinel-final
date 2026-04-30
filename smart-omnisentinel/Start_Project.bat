@echo off
cd /d "%~dp0"
color 0A
title SmartOmniSentinel Project Launcher

echo ========================================================
echo         SMART OMNI SENTINEL - PROJECT LAUNCHER
echo ========================================================
echo.
echo Starting FastAPI Backend Server (Port 8000)...
start "SmartOmniSentinel Backend" cmd /c "uvicorn api.main:app --host 0.0.0.0 --port 8000"

echo Starting Dashboard Web Server (Port 3000)...
start "SmartOmniSentinel Dashboard" cmd /c "python -m http.server 3000 --directory dashboard"

echo.
echo Servers started successfully!
echo Waiting 3 seconds for backend to initialize...
timeout /t 3 /nobreak >nul

echo Opening browser to http://localhost:3000...
start http://localhost:3000

echo.
echo ========================================================
echo  Step 1: Make sure DroidCam is running on your phone.
echo  Step 2: Connect phone and laptop to same Wi-Fi.
echo  Step 3: Keep the browser window visible.
echo ========================================================
echo.
:: Hardcoded DroidCam IP address
set DROID_IP=192.168.137.150

echo Starting AI Camera Engine (Mobile Camera Ingest)...
::start "SmartOmniSentinel AI (Mobile)" cmd /k "python scripts/demo_ingest.py --video http://%DROID_IP%:4747/video --camera "Main Entrance""

echo Starting AI Camera Engine (Laptop Webcam Ingest)...
::start "SmartOmniSentinel AI (Laptop)" cmd /k "python scripts/demo_ingest.py --video 0 --camera "Parking Lot B""

:: ============================================================================== 
:: HOW TO RUN THE SKELETON VIEW (demo_live.py) INSTEAD
:: ==============================================================================
:: A camera can only be opened by one script at a time.
:: 
:: If you want to show the judges the live AI Skeleton tracking popup window, 
:: ADD a double-colon (::) in front of the demo_ingest.py line above to mute it.
:: Then, REMOVE the double-colon from one of the demo_live.py lines below:
::
:: For Mobile Camera Skeleton Demo:
:: python scripts/demo_live.py --video "http://%DROID_IP%:4747/video"
::
:: For Laptop Webcam Skeleton Demo (Uncomment line below):
python scripts/demo_live.py --video 0
:: ==============================================================================

pause
 