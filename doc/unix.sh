#!/bin/bash

PWD_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# module map
declare -A MODULES=(
    ["gwas"]="gwas.py"
    ["test"]="test.py"
)

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo -e "\\e[32mUsage: $0 <module> [parameter]\\e[0m"
    echo -e "\\e[32mModule:\\e[0m"
    for module in "${!MODULES[@]}"; do
        echo -e "\\e[32m  $module   execute ${MODULES[$module]}\\e[0m"
    done
    exit 0
fi

MODULE_NAME="$1"
shift  # remove parameter of module name

if [[ -n "${MODULES[$MODULE_NAME]}" ]]; then
    SCRIPT_FILE="${MODULES[$MODULE_NAME]}"
    .venv/Scripts/python -u "$PWD_DIR/module/$SCRIPT_FILE" "$@"
else
    echo -e "\\e[31mError: unknown module '$MODULE_NAME'!\\e[0m"
    echo -e "\\e[31mModule: ${!MODULES[*]}\\e[0m"
    exit 1
fi