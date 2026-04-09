@echo off
setlocal

set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..") do set REPO_ROOT=%%~fI

if defined CHEK_EGO_MINER_PYTHON (
  set PYTHON_CMD=%CHEK_EGO_MINER_PYTHON%
) else (
  where py >nul 2>nul
  if %ERRORLEVEL%==0 (
    set PYTHON_CMD=py -3
  ) else (
    set PYTHON_CMD=python
  )
)

if defined PYTHONPATH (
  set PYTHONPATH=%REPO_ROOT%\cli;%REPO_ROOT%;%PYTHONPATH%
) else (
  set PYTHONPATH=%REPO_ROOT%\cli;%REPO_ROOT%
)

%PYTHON_CMD% -m chek_ego_miner.main %*
