pip install uv
uv venv
uv sync
uv pip install -r gwas.requirements.txt
cp ./doc/unix.sh GWAS.sh
chmod +x GWAS.sh