#!/bin/bash
# installs the python virtual environment

python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r "install/python_dependencies.txt"
deactivate