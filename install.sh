echo "Process of install..."
PWD_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pip install uv
uv venv --clear
uv pip install -r gwas.requirements.txt
cp ./doc/unix.sh GWAS
chmod +x GWAS
echo "Completed! please add $PWD_DIR to your enviroment"
echo "echo "export PATH="$PWD_DIR:\$PATH"" >> ~/.bashrc; source ~/.bashrc"