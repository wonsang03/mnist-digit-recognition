@echo off
REM Created: 2025-09-15
REM Double-click this file to train the model (first run only), start the Flask
REM server and open the web handwritten digit recognizer in a browser.

setlocal
cd /d "%~dp0"
title Handwritten Digit Recognition - Web

echo ============================================================
echo   Handwritten Digit Recognition - Web Version
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo Install Python 3.10 or newer and tick "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

if not exist "model\mnist_cnn.pt" (
    echo [1/3] No trained model found. Training on MNIST now.
    echo       The first run downloads the dataset and takes a few minutes.
    echo.
    python train_model.py
    if errorlevel 1 (
        echo.
        echo [ERROR] Training failed. Install the requirements first:
        echo         pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Trained model found. Skipping training.
)

echo.
echo [2/3] Opening http://localhost:5000 in your browser...
start "" "http://localhost:5000"

echo [3/3] Starting the Flask server. Press Ctrl+C to stop it.
echo.
python web_version\app.py
if errorlevel 1 (
    echo.
    echo [ERROR] The server exited with an error.
    pause
    exit /b 1
)

endlocal
