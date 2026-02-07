@echo off
REM Training launcher script for Windows
REM This script applies memory optimizations before training

echo ================================================================================
echo PIANO TRAINING - WINDOWS MEMORY OPTIMIZED LAUNCHER
echo ================================================================================
echo.

REM Check if python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please ensure Python is installed and added to PATH
    pause
    exit /b 1
)

echo Checking memory status...
python memory_optimizations.py
if errorlevel 1 (
    echo ERROR: Memory check failed
    pause
    exit /b 1
)

echo.
echo Starting training...
echo ================================================================================
echo.

REM Run training with optimizations
python 1_train_onsets_velocities.py %*

pause
