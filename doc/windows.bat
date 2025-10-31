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
    
    rem 手动构建新的参数列表
    set "NEW_ARGS="
    :build_args
    if "%~1"=="" goto execute_script
    if defined NEW_ARGS (
        set "NEW_ARGS=!NEW_ARGS! %1"
    ) else (
        set "NEW_ARGS=%1"
    )
    shift
    goto build_args

    :execute_script
    call %PWD_DIR%\.venv\Scripts\python -u "%PWD_DIR%\module\!SCRIPT_FILE!" !NEW_ARGS!
)

exit /b 0

:show_help
echo Usage: %0 ^<module^> [parameter]
echo Module:
echo   gwas   execute gwas.py
echo   test   execute test.py
exit /b 0