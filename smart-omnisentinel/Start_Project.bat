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
start "SmartOmniSentinel Backend" cmd /k ".\venv\Scripts\uvicorn.exe api.main:app --host 0.0.0.0 --port 8000"

echo Starting Dashboard Web Server (Port 3000)...
start "SmartOmniSentinel Dashboard" cmd /k ".\venv\Scripts\python.exe -m http.server 3000 --directory dashboard"

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
echo   SELECT SURVEILLANCE OPERATIONS MODE:
echo ========================================================
echo   [1] FULL TRIPLE-CCTV CONTROL ROOM (Recommended)
echo       - CAM-01: DroidCam Mobile Phone (Live WiFi/USB)
echo       - CAM-02: Laptop Webcam (Built-in Camera)
echo       - CAM-03: Stairwell C3 (Non-Violent Video Loop)
echo.
echo   [2] LAPTOP WEBCAM + STAIRWELL VIDEO ONLY (No phone)
echo       - CAM-01: Demo Fight Video (real_fight.mp4)
echo       - CAM-02: Laptop Webcam (Live Camera 0)
echo       - CAM-03: Stairwell C3 (Non-Violent Video Loop)
echo.
echo   [3] DROIDCAM PHONE ONLY
echo       - CAM-01: DroidCam Mobile Phone (Live)
echo       - CAM-02: Laptop Webcam (Live Camera 0)
echo.
echo   [4] ALL-VIDEO SIMULATION MODE (No webcams needed)
echo       - CAM-01: real_fight.mp4
echo       - CAM-02: real_fight1.mp4
echo       - CAM-03: cam_stairwell.mp4
echo ========================================================
set /p SOURCE_CHOICE="Select option [1, 2, 3, or 4, default: 1]: "
if "%SOURCE_CHOICE%"=="" set SOURCE_CHOICE=1

if "%SOURCE_CHOICE%"=="2" (
    echo.
    echo Starting CAM-01: Demo Fight Video (real_fight.mp4)...
    start "SmartOmniSentinel AI CAM-01 (Video)" cmd /k .\venv\Scripts\python.exe scripts/demo_ingest.py --video real_fight.mp4 --camera "Main Entrance" --live_frame dashboard/cam_mobile.jpg --engine pose

    echo Starting CAM-02: Laptop Webcam (Built-in Camera 0)...
    start "SmartOmniSentinel AI CAM-02 (Laptop)" cmd /k .\venv\Scripts\python.exe scripts/demo_ingest.py --video 0 --camera "Parking Lot B" --live_frame dashboard/cam_laptop.jpg --engine pose

    echo Starting CAM-03: Stairwell C3 Video (Normal Non-Violent Loop)...
    start "SmartOmniSentinel AI CAM-03 (Stairwell)" cmd /k .\venv\Scripts\python.exe scripts/demo_ingest.py --video dashboard/cam_stairwell.mp4 --camera "Stairwell C3" --live_frame dashboard/cam_stairwell.jpg --engine pose

) else if "%SOURCE_CHOICE%"=="3" (
    set DEFAULT_IP=192.168.137.120
    echo.
    echo Enter the WiFi IP Address shown in your DroidCam phone app:
    set /p DROID_IP="IP Address [default: %DEFAULT_IP%]: "
    if "%DROID_IP%"=="" set DROID_IP=%DEFAULT_IP%

    echo Starting CAM-01: DroidCam Mobile Camera (http://%DROID_IP%:4747/video)...
    start "SmartOmniSentinel AI CAM-01 (Mobile)" cmd /k .\venv\Scripts\python.exe scripts/demo_ingest.py --video http://%DROID_IP%:4747/video --camera "Main Entrance" --live_frame dashboard/cam_mobile.jpg --engine pose

    echo Starting CAM-02: Laptop Webcam (Built-in Camera 0)...
    start "SmartOmniSentinel AI CAM-02 (Laptop)" cmd /k .\venv\Scripts\python.exe scripts/demo_ingest.py --video 0 --camera "Parking Lot B" --live_frame dashboard/cam_laptop.jpg --engine pose

) else if "%SOURCE_CHOICE%"=="4" (
    echo.
    echo Starting CAM-01: real_fight.mp4...
    start "SmartOmniSentinel AI CAM-01 (Video)" cmd /k .\venv\Scripts\python.exe scripts/demo_ingest.py --video real_fight.mp4 --camera "Main Entrance" --live_frame dashboard/cam_mobile.jpg --engine pose

    echo Starting CAM-02: real_fight1.mp4...
    start "SmartOmniSentinel AI CAM-02 (Video)" cmd /k .\venv\Scripts\python.exe scripts/demo_ingest.py --video real_fight1.mp4 --camera "Parking Lot B" --live_frame dashboard/cam_laptop.jpg --engine pose

    echo Starting CAM-03: cam_stairwell.mp4...
    start "SmartOmniSentinel AI CAM-03 (Stairwell)" cmd /k .\venv\Scripts\python.exe scripts/demo_ingest.py --video dashboard/cam_stairwell.mp4 --camera "Stairwell C3" --live_frame dashboard/cam_stairwell.jpg --engine pose

) else (
    REM Default: Option 1 (All 3 Live Feeds)
    set DEFAULT_IP=192.168.137.120
    echo.
    echo Enter the WiFi IP Address shown in your DroidCam phone app:
    set /p DROID_IP="IP Address [default: %DEFAULT_IP%]: "
    if "%DROID_IP%"=="" set DROID_IP=%DEFAULT_IP%

    echo.
    echo Starting CAM-01: DroidCam Mobile Camera (http://%DROID_IP%:4747/video)...
    start "SmartOmniSentinel AI CAM-01 (Mobile)" cmd /k .\venv\Scripts\python.exe scripts/demo_ingest.py --video http://%DROID_IP%:4747/video --camera "Main Entrance" --live_frame dashboard/cam_mobile.jpg --engine pose

    echo Starting CAM-02: Laptop Webcam (Built-in Camera 0)...
    start "SmartOmniSentinel AI CAM-02 (Laptop)" cmd /k .\venv\Scripts\python.exe scripts/demo_ingest.py --video 0 --camera "Parking Lot B" --live_frame dashboard/cam_laptop.jpg --engine pose

    echo Starting CAM-03: Stairwell C3 Video (Normal Non-Violent Loop)...
    start "SmartOmniSentinel AI CAM-03 (Stairwell)" cmd /k .\venv\Scripts\python.exe scripts/demo_ingest.py --video dashboard/cam_stairwell.mp4 --camera "Stairwell C3" --live_frame dashboard/cam_stairwell.jpg --engine pose
)

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
 