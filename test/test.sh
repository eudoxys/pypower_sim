#!/bin/bash
test -d .venv || python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install --upgrade -r ../requirements.txt
pip install --upgrade ..
python3 test.py