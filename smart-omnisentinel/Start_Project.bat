@echo off
cd /d "%~dp0"

color 0A
title SmartOmniSentinel V2 - Security Operations Center

cls
echo ========================================================
echo   SMART OMNISENTINEL V2 - AI SURVEILLANCE PLATFORM
echo   ST-GCN + GRU (74.8%% Acc) ^| Proximity ^| Weapons
echo   Optimized for Intel Core Ultra 9 / CPU Edge Ingestion
echo ========================================================
echo.
echo   SELECT SURVEILLANCE OPERATION MODE:
echo.
echo   [1] FULL TRIPLE-CCTV CONTROL ROOM (Recommended)
echo       - CAM-01: DroidCam Mobile Phone (Live WiFi/USB)
echo       - CAM-02: Laptop Webcam (Live Camera 0)
echo       - CAM-03: Stairwell C3 (Non-Violent Video Loop)
echo.
echo   [2] LAPTOP WEBCAM + STAIRWELL VIDEO (No phone needed)
echo       - CAM-01: Demo Fight Video (real_fight.mp4)
echo       - CAM-02: Laptop Webcam (Live Camera 0)
echo       - CAM-03: Stairwell C3 (Non-Violent Video Loop)
echo.
echo   [3] DROIDCAM PHONE ONLY
echo       - CAM-01: DroidCam Mobile Phone (Live)
echo       - CAM-02: Laptop Webcam (Live Camera 0)
echo.
echo   [4] ALL-VIDEO SIMULATION MODE (Zero webcams needed)
echo       - CAM-01: real_fight.mp4
echo       - CAM-02: real_fight1.mp4
echo       - CAM-03: cam_stairwell.mp4
echo.
echo   [5] SKELETON HUD VIEWER (Standalone GUI Window)
echo       - Real-time 17-keypoint bone tracking + proximity lines
echo ========================================================
choice /c 12345 /n /m "Select mode [1, 2, 3, 4, or 5]: "

if errorlevel 5 goto MENU_SKELETON
if errorlevel 4 goto SETUP_MODE_4
if errorlevel 3 goto SETUP_MODE_3
if errorlevel 2 goto SETUP_MODE_2
if errorlevel 1 goto SETUP_MODE_1

:SETUP_MODE_1
set MODE_NAME=1
set DROID_IP=10.92.36.164
echo.
echo Enter the WiFi IP Address shown in your DroidCam phone app:
set /p USER_IP="IP Address [press Enter for 10.92.36.164]: "
if "%USER_IP%"=="" set USER_IP=10.92.36.164
set USER_IP=%USER_IP: =%
if "%USER_IP%"=="" set USER_IP=10.92.36.164
set DROID_IP=%USER_IP%
goto START_SERVERS

:SETUP_MODE_2
set MODE_NAME=2
goto START_SERVERS

:SETUP_MODE_3
set MODE_NAME=3
set DROID_IP=10.92.36.164
echo.
echo Enter the WiFi IP Address shown in your DroidCam phone app:
set /p USER_IP="IP Address [press Enter for 10.92.36.164]: "
if "%USER_IP%"=="" set USER_IP=10.92.36.164
set USER_IP=%USER_IP: =%
if "%USER_IP%"=="" set USER_IP=10.92.36.164
set DROID_IP=%USER_IP%
goto START_SERVERS

:SETUP_MODE_4
set MODE_NAME=4
goto START_SERVERS

:MENU_SKELETON
echo.
echo ========================================================
echo   SKELETON HUD VIEWER - SELECT CAMERA INPUT:
echo   [1] DroidCam Mobile Phone Camera
echo   [2] Laptop Built-in Webcam (Camera 0)
echo   [3] Demo Fight Video File (real_fight.mp4)
echo ========================================================
choice /c 123 /n /m "Select input [1, 2, or 3]: "

if errorlevel 3 goto LAUNCH_SKEL_VIDEO
if errorlevel 2 goto LAUNCH_SKEL_LAPTOP
if errorlevel 1 goto LAUNCH_SKEL_PHONE

:LAUNCH_SKEL_PHONE
set DROID_IP=10.92.36.164
echo.
echo Enter the WiFi IP Address shown in your DroidCam phone app:
set /p USER_IP="IP Address [press Enter for 10.92.36.164]: "
if "%USER_IP%"=="" set USER_IP=10.92.36.164
set USER_IP=%USER_IP: =%
if "%USER_IP%"=="" set USER_IP=10.92.36.164
set DROID_IP=%USER_IP%
echo.
echo Launching Skeleton HUD on Mobile Phone: http://%DROID_IP%:4747/video
.\venv\Scripts\python.exe scripts\live_skeleton_window.py --source http://%DROID_IP%:4747/video
goto SCRIPT_EXIT

:LAUNCH_SKEL_LAPTOP
echo.
echo Launching Skeleton HUD on Laptop Webcam (Camera 0)...
.\venv\Scripts\python.exe scripts\live_skeleton_window.py --source 0
goto SCRIPT_EXIT

:LAUNCH_SKEL_VIDEO
echo.
echo Launching Skeleton HUD on Demo Video (real_fight.mp4)...
.\venv\Scripts\python.exe scripts\live_skeleton_window.py --source real_fight.mp4
goto SCRIPT_EXIT

:START_SERVERS
echo.
echo ========================================================
echo Starting Core Servers and V2 AI Ingestion Pipelines...
echo ========================================================

:: 1. Start FastAPI Backend (Port 8000) - Minimized
echo [1/4] Starting FastAPI Backend (Port 8000)...
start /min "SmartOmniSentinel Backend" cmd /k .\venv\Scripts\uvicorn.exe api.main:app --host 0.0.0.0 --port 8000

:: 2. Start Dashboard Web Server (Port 3000) - Minimized
echo [2/4] Starting Dashboard Web Server (Port 3000)...
start /min "SmartOmniSentinel Dashboard" cmd /k .\venv\Scripts\python.exe -m http.server 3000 --directory dashboard

echo Waiting 3 seconds for backend initialization...
ping 127.0.0.1 -n 4 >nul

:: 3. Start Camera Ingestion Pipelines based on choice
echo [3/4] Launching V2 AI Ingestion Pipelines (ST-GCN + YOLOv8-Pose)...
if "%MODE_NAME%"=="4" goto RUN_CAMERAS_MODE_4
if "%MODE_NAME%"=="3" goto RUN_CAMERAS_MODE_3
if "%MODE_NAME%"=="2" goto RUN_CAMERAS_MODE_2
goto RUN_CAMERAS_MODE_1

:RUN_CAMERAS_MODE_1
start /min "CAM-01 (Mobile)" cmd /k .\venv\Scripts\python.exe scripts\demo_ingest.py --video http://%DROID_IP%:4747/video --camera Main_Entrance --live_frame dashboard\cam_mobile.jpg --engine pose
start /min "CAM-02 (Laptop)" cmd /k .\venv\Scripts\python.exe scripts\demo_ingest.py --video 0 --camera Parking_Lot_B --live_frame dashboard\cam_laptop.jpg --engine pose
start /min "CAM-03 (Stairwell)" cmd /k .\venv\Scripts\python.exe scripts\demo_ingest.py --video dashboard\cam_stairwell.mp4 --camera Stairwell_C3 --live_frame dashboard\cam_stairwell.jpg --engine pose
goto FINISH_LAUNCH

:RUN_CAMERAS_MODE_2
start /min "CAM-01 (Video)" cmd /k .\venv\Scripts\python.exe scripts\demo_ingest.py --video real_fight.mp4 --camera Main_Entrance --live_frame dashboard\cam_mobile.jpg --engine pose
start /min "CAM-02 (Laptop)" cmd /k .\venv\Scripts\python.exe scripts\demo_ingest.py --video 0 --camera Parking_Lot_B --live_frame dashboard\cam_laptop.jpg --engine pose
start /min "CAM-03 (Stairwell)" cmd /k .\venv\Scripts\python.exe scripts\demo_ingest.py --video dashboard\cam_stairwell.mp4 --camera Stairwell_C3 --live_frame dashboard\cam_stairwell.jpg --engine pose
goto FINISH_LAUNCH

:RUN_CAMERAS_MODE_3
start /min "CAM-01 (Mobile)" cmd /k .\venv\Scripts\python.exe scripts\demo_ingest.py --video http://%DROID_IP%:4747/video --camera Main_Entrance --live_frame dashboard\cam_mobile.jpg --engine pose
start /min "CAM-02 (Laptop)" cmd /k .\venv\Scripts\python.exe scripts\demo_ingest.py --video 0 --camera Parking_Lot_B --live_frame dashboard\cam_laptop.jpg --engine pose
goto FINISH_LAUNCH

:RUN_CAMERAS_MODE_4
start /min "CAM-01 (Video)" cmd /k .\venv\Scripts\python.exe scripts\demo_ingest.py --video real_fight.mp4 --camera Main_Entrance --live_frame dashboard\cam_mobile.jpg --engine pose
start /min "CAM-02 (Video)" cmd /k .\venv\Scripts\python.exe scripts\demo_ingest.py --video real_fight1.mp4 --camera Parking_Lot_B --live_frame dashboard\cam_laptop.jpg --engine pose
start /min "CAM-03 (Stairwell)" cmd /k .\venv\Scripts\python.exe scripts\demo_ingest.py --video dashboard\cam_stairwell.mp4 --camera Stairwell_C3 --live_frame dashboard\cam_stairwell.jpg --engine pose
goto FINISH_LAUNCH

:FINISH_LAUNCH
:: 4. Open Browser
echo [4/4] Opening Security Operations Dashboard...
start http://localhost:3000

echo.
echo ========================================================
echo   SMART OMNISENTINEL V2 IS NOW LIVE!
echo ========================================================
echo   Dashboard URL : http://localhost:3000
echo   API Backend   : http://localhost:8000/docs
echo.
echo   All services and cameras are running smoothly in background.
echo   To stop all Sentinel processes, close this window or run:
echo   taskkill /f /im python.exe
echo ========================================================
echo.
pause

:SCRIPT_EXIT
