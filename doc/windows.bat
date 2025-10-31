@echo off
setlocal EnableDelayedExpansion

set "PWD_DIR=%~dp0"

rem Remove trailing backslash if present
if "%PWD_DIR:~-1%"=="\" set "PWD_DIR=%PWD_DIR:~0,-1%"

rem Module map - using variables to simulate associative array
set "MODULE_gwas=gwas.py"
set "MODULE_test=test.py"

if "%1"=="" goto show_help
if "%1"=="-h" goto show_help
if "%1"=="--help" goto show_help

set "MODULE_NAME=%1"
shift

rem Check if module exists and execute corresponding script
if defined MODULE_%MODULE_NAME% (
    set "SCRIPT_FILE=!MODULE_%MODULE_NAME%!"
    call .venv\Scripts\python -u "%PWD_DIR%\module\!SCRIPT_FILE!" %*
) else (
    echo Error: unknown module '%MODULE_NAME%'!
    echo Available modules: gwas test
    exit /b 1
)

exit /b 0

:show_help
echo Usage: %0 ^<module^> [parameter]
echo Module:
echo   gwas   execute gwas.py
echo   test   execute test.py
exit /b 0