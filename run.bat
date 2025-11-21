@echo off
REM Wrapper script for profanity_filter.py with CUDA/cuDNN support (Windows)

set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%venv

REM Add cuDNN DLLs to PATH for GPU support
set PATH=%VENV_DIR%\Lib\site-packages\nvidia\cudnn\bin;%VENV_DIR%\Lib\site-packages\nvidia\cublas\bin;%PATH%

REM Run the profanity filter
"%VENV_DIR%\Scripts\python.exe" "%SCRIPT_DIR%profanity_filter.py" %*
