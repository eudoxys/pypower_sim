#!/bin/bash
test -d .venv || python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip 1>/dev/null
pip install --upgrade -r ../requirements.txt 1>/dev/null
pip install --upgrade -e .. 1>/dev/null
python3 test.py
cd ../pypower_sim
for pyfile in [a-z]*.py; do
    python3 $pyfile
done
