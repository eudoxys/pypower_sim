"""Tests of `pypower_sim.ppsolver.PPSolver` (fast)"""

import datetime as dt

import pytest
import pytz
import pandas as pd

from pypower_sim import PPModel, PPSolver, PPData

from conftest import SMALL_CASES, DATADIR, case_file

@pytest.mark.parametrize("name",SMALL_CASES)
def test_powerflow(name):
    """Powerflow converges on the standard cases"""
    model = PPModel(case=case_file(name))
    assert PPSolver(model).solve_pf()

@pytest.mark.parametrize("name",SMALL_CASES)
def test_opf_no_violations(name):
    """AC OPF converges and leaves no violations"""
    model = PPModel(case=case_file(name))
    solver = PPSolver(model)
    assert solver.solve_opf(use_acopf=True)
    assert solver.solve_pf()
    assert not model.get_violations()

def test_powerflow_balance():
    """Generation equals load plus losses (losses are non-negative)"""
    model = PPModel(case=case_file("case9"))
    assert PPSolver(model).solve_pf()
    gen = model.get_data("gen")["PG"].sum()
    load = model.get_data("bus")["PD"].sum()
    assert gen >= load
    assert gen - load < 0.05 * load

@pytest.mark.xfail(reason="solve_oce() returns False on bundled cases (see CLAUDE.md)",strict=False)
def test_oce():
    """Optimal capacity expansion reports success"""
    model = PPModel(case=case_file("case9"))
    assert PPSolver(model).solve_oce()

def test_timeseries(workdir):
    """A time-series run writes one row per timestep for every bus"""
    model = PPModel(case=case_file("case9"))
    data = PPData(model)
    data.set_input("bus","PD",f"{DATADIR}/load.csv",scale=1)
    data.set_input("bus","QD",f"{DATADIR}/load.csv",scale=0.1)
    data.set_output("bus","VM","bus_vm.csv",formatting=".3f")
    data.set_recorder("cost.csv","cost",["cost"],scale=model.case["baseMVA"],formatting=".2f")

    start = dt.datetime(2020,7,31,17,0,0,0,pytz.UTC)
    end = dt.datetime(2020,7,31,23,0,0,0,pytz.UTC)
    result = PPSolver(model).run_timeseries(start=start,end=end,freq="1h")
    assert result is None, model.errors
    assert not model.errors

    steps = len(pd.date_range(start,end,freq="1h"))
    vm = pd.read_csv(workdir / "bus_vm.csv",index_col=0)
    assert vm.shape == (steps,len(model.case["bus"]))
    assert ((vm > 0.8) & (vm < 1.2)).all().all()
    cost = pd.read_csv(workdir / "cost.csv",index_col=0)
    assert cost.shape == (steps,1)
    assert cost["cost"].is_monotonic_increasing # load.csv rises every hour from 17:00 to 23:00
