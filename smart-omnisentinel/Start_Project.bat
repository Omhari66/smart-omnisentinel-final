@echo off
cd /d "%~dp0"
set OLLAMA_MODELS=%~dp0models\ollama
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

echo Starting Local Ollama LLM Server (Port 11434)...
start "SmartOmniSentinel Ollama LLM" cmd /c "set OLLAMA_MODELS=%~dp0models\ollama && ollama serve"

echo.
echo Servers started successfully!
echo Waiting 5 seconds for backend to initialize...
timeout /t 5 /nobreak >nul

echo Opening browser to http://localhost:3000...
start http://localhost:3000

echo.
echo ========================================================
echo Select Video Input Source for Main Entrance:
echo [1] DroidCam Mobile Phone (Live Stream)
echo [2] Demo Video File (your_video.mp4)
echo [3] Real Fight Video File (real_fight.mp4)
echo ========================================================
set /p SOURCE_CHOICE="Select option [1, 2, or 3, default: 1]: "

if "%SOURCE_CHOICE%"=="2" (
    echo Starting AI Camera Engine - Video File Ingest - using your_video.mp4...
    start "SmartOmniSentinel AI (Video)" cmd /k python scripts/demo_ingest.py --video your_video.mp4 --camera "Main Entrance" --live_frame dashboard/cam_mobile.jpg --engine pose
) else if "%SOURCE_CHOICE%"=="3" (
    echo Starting AI Camera Engine - Video File Ingest - using real_fight.mp4...
    start "SmartOmniSentinel AI (Video)" cmd /k python scripts/demo_ingest.py --video real_fight.mp4 --camera "Main Entrance" --live_frame dashboard/cam_mobile.jpg --engine pose
) else (
    REM DroidCam IP address configuration
    set DEFAULT_IP=192.168.137.120
    echo.
    echo Please enter the DroidCam IP Address shown on your phone's DroidCam app.
    set /p DROID_IP="IP Address [default: %DEFAULT_IP%]: "
    if "%DROID_IP%"=="" set DROID_IP=%DEFAULT_IP%

    echo Starting AI Camera Engine - Mobile Camera Ingest - using http://%DROID_IP%:4747/video...
    start "SmartOmniSentinel AI (Mobile)" cmd /k python scripts/demo_ingest.py --video http://%DROID_IP%:4747/video --camera "Main Entrance" --live_frame dashboard/cam_mobile.jpg --engine pose
)

echo Starting AI Camera Engine (Laptop Webcam Ingest)...
::start "SmartOmniSentinel AI (Laptop)" cmd /k "python scripts/demo_ingest.py --video 0 --camera "Parking Lot B" --live_frame dashboard/cam_laptop.jpg --engine pose"

:: ============================================================================== 
:: HOW TO SWITCH AI ENGINES (MODELS)
:: ==============================================================================
:: You can now easily switch between the two models by changing "--engine pose" above:
:: 
:: Option 1: "--engine pose" (DEFAULT)
::   Uses YOLOv8 + GRU. Zero lag. Perfect for edge devices. Accurate snapshots.
::
:: Option 2: "--engine cnn"
::   Uses the heavy RWF-2000 3D CNN. Highly accurate, but requires a powerful GPU.
::   If your system lags, it will skip frames and perform poorly.
:: ==============================================================================

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
::python scripts/demo_live.py --video "http://%DROID_IP%:4747/video"
::
::For Laptop Webcam Skeleton Demo (Uncomment line below):
::python scripts/demo_live.py --video 0
:: ==============================================================================

pause
 