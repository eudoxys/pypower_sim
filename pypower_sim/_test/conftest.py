"""Shared test setup

Test inputs are in `_test/data/`. Tests that write files run in a pytest
temporary folder (see the `workdir` fixture), never in the repository.
"""

import os

import pytest

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data")
"""Folder holding the test cases and input data"""

SMALL_CASES = ["case4gsm","case6ww","case9","case9Q","case14","case30","case39","case57"]
"""Standard `pypower` cases small enough to solve in well under a second"""

WECC_CASES = ["case240_2011m","case240_2018m","case240_2020m"]
"""WECC 240-bus cases (slow)"""

def case_file(name:str) -> str:
    """Path to a test case file"""
    return os.path.join(DATADIR,f"{name}.py")

@pytest.fixture
def workdir(tmp_path,monkeypatch):
    """Run the test in a temporary folder"""
    monkeypatch.chdir(tmp_path)
    return tmp_path
