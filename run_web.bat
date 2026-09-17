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
    echo [1/2] No trained model found. Training on MNIST now.
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
    echo [1/2] Trained model found. Skipping training.
)

echo.
echo [2/2] Starting the Flask server. The browser opens once it is ready.
echo       Press Ctrl+C to stop it.
echo.
REM The server opens the browser itself: a browser launched from here raced
REM ahead of model loading and landed on a connection-refused page.
python web_version\app.py --open-browser
if errorlevel 1 (
    echo.
    echo [ERROR] The server exited with an error.
    pause
    exit /b 1
)

endlocal
