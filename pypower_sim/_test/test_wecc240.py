"""Tests on the WECC 240-bus cases (slow, ~10 s each)

Run alone with `pytest -m slow`; skip with `pytest -m "not slow"`.
"""

import pytest
import pandas as pd

from pypower_sim import PPModel, PPSolver, PPGIS

from conftest import WECC_CASES, DATADIR, case_file

pytestmark = pytest.mark.slow

@pytest.mark.parametrize("name",WECC_CASES)
def test_powerflow(name):
    """Powerflow converges with no violations"""
    model = PPModel(case=case_file(name))
    assert len(model.case["bus"]) == 243
    assert PPSolver(model).solve_pf()
    assert not model.get_violations()

@pytest.mark.parametrize("name",[
    "case240_2011m",
    pytest.param("case240_2018m",marks=pytest.mark.xfail(reason="AC OPF does not converge",strict=False)),
    "case240_2020m",
    ])
def test_acopf(name):
    """AC OPF converges"""
    model = PPModel(case=case_file(name))
    assert PPSolver(model).solve_opf(use_acopf=True)

def test_gis():
    """GIS data attaches to every bus"""
    model = PPModel(case=case_file("case240_2020m"))
    PPGIS(model,pd.read_csv(f"{DATADIR}/case240_gis.csv"))
    assert model.has("gis")
    assert len(model.get_data("gis")) == len(model.case["bus"])
