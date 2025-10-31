pip install uv
uv venv --clear
uv sync
uv pip install -r gwas.requirements.txt
copy .\doc\windows.bat GWAS.bat