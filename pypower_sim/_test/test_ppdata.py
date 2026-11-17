"""Tests of `pypower_sim.ppdata.PPData` input panels and players (fast)"""

import datetime as dt

import pytest
import pytz
import numpy as np
import pandas as pd

from pypower_sim import PPModel, PPSolver, PPData

from conftest import DATADIR, case_file

T0 = pd.Timestamp("2020-07-31 17:00:00+00:00")
T1 = pd.Timestamp("2020-07-31 18:00:00+00:00")

def write_player(path,columns:list[str],rows:list[list[float]]) -> str:
    """Write a two-row player CSV and return its name"""
    with open(path,"w",encoding="utf-8") as fh:
        print("datetime",*columns,sep=",",file=fh)
        for t,row in zip([T0,T1],rows):
            print(t.isoformat(),*row,sep=",",file=fh)
    return str(path)

@pytest.fixture(name="model")
def fixture_model():
    """Fresh case9 model"""
    return PPModel(case=case_file("case9"))

def rate(model,row,prop="RATE_A"):
    """Branch property value"""
    return model.get_data("branch").loc[row,prop]

def test_input_panel(model):
    """An input panel sets one property of the objects named by its columns"""
    PPData(model).set_input("bus","PD",f"{DATADIR}/load.csv",scale=2)
    assert PPSolver(model).update_inputs(T0) == 0
    assert list(model.get_data("bus").loc[[1,2,3,6],"PD"]) == [20,40,60,80]

def test_input_remove(model):
    """`file=None` removes one input, or all inputs of a table with `column=None`"""
    data = PPData(model)
    data.set_input("bus","PD",f"{DATADIR}/load.csv")
    data.set_input("bus","QD",f"{DATADIR}/load.csv")
    data.set_input("gen","PG",f"{DATADIR}/load.csv")
    data.set_input("bus","PD",None)
    assert set(model.inputs) == {("bus","QD"),("gen","PG")}
    with pytest.raises(AssertionError,match="not defined"):
        data.set_input("bus","PD",None)
    data.set_input("bus","PD",f"{DATADIR}/load.csv") # can be set again
    data.set_input("bus",None,None)
    assert set(model.inputs) == {("gen","PG")}
    data.set_input("bus",None,None) # nothing left to remove
    with pytest.raises(AssertionError,match="not found"):
        data.set_input("bus",None,f"{DATADIR}/load.csv")

def test_player_parent(model,workdir):
    """Plain column names are properties of the parent object"""
    file = write_player(workdir/"p.csv",["RATE_A","RATE_B"],[[100,110],[200,210]])
    PPData(model).set_player("branch",file,parent=3)
    solver = PPSolver(model)
    assert solver.update_players(T0) == 0
    assert (rate(model,3),rate(model,3,"RATE_B")) == (100,110)
    assert solver.update_players(T1) == 0
    assert (rate(model,3),rate(model,3,"RATE_B")) == (200,210)

def test_player_objects(model,workdir):
    """`OBJECTNAME.PROPERTYNAME` overrides the parent; no parent needed"""
    before = rate(model,0)
    file = write_player(workdir/"p.csv",["4.RATE_A","5.RATE_C"],[[1,2],[3,4]])
    PPData(model).set_player("branch",file)
    assert PPSolver(model).update_players(T1) == 0
    assert (rate(model,4),rate(model,5,"RATE_C"),rate(model,0)) == (3,4,before)

def test_player_mixed_scale(model,workdir):
    """Parent and object columns combine; scale and offset apply to all"""
    file = write_player(workdir/"p.csv",["RATE_A","2.RATE_A"],[[10,20],[30,40]])
    PPData(model).set_player("branch",file,parent=1,scale=2,offset=1)
    assert PPSolver(model).update_players(T0) == 0
    assert (rate(model,1),rate(model,2)) == (21,41)

def test_player_mapping(model,workdir):
    """Object names map to rows by a data column or a dict"""
    file = write_player(workdir/"p.csv",["PMAX","3.QMAX"],[[111,222],[0,0]])
    PPData(model).set_player("gen",file,parent=2,mapping=("GEN_BUS",int))
    assert PPSolver(model).update_players(T0) == 0
    gen = model.get_data("gen").set_index("GEN_BUS")
    assert (gen.loc[2,"PMAX"],gen.loc[3,"QMAX"]) == (111,222)

    file = write_player(workdir/"q.csv",["RATE_A","line.b.RATE_A"],[[5,6],[0,0]])
    PPData(model).set_player("branch",file,parent="line.a",mapping={"line.a":7,"line.b":8})
    assert PPSolver(model).update_players(T0) == 0
    assert (rate(model,7),rate(model,8)) == (5,6)

def test_player_parts(model,workdir):
    """`.PROPERTYNAME.PART` refers to the parent; `real` works on real data"""
    columns = [".RATE_A.real",".RATE_B","6.RATE_C.real"]
    file = write_player(workdir/"p.csv",columns,[[7,8,9],[0,0,0]])
    PPData(model).set_player("branch",file,parent=0)
    assert PPSolver(model).update_players(T0) == 0
    assert (rate(model,0),rate(model,0,"RATE_B"),rate(model,6,"RATE_C")) == (7,8,9)

def test_player_imag(model,workdir):
    """`imag` sets the imaginary part of complex data only"""
    file = write_player(workdir/"p.csv",[".PD.imag",".PD.real"],[[3,4],[0,0]])
    with pytest.raises(AssertionError,match="imaginary"):
        PPData(model).set_player("bus",file,parent=4)
    model.case["bus"] = model.case["bus"].astype(complex)
    PPData(model).set_player("bus",file,parent=4)
    assert PPSolver(model).update_players(T0) == 0
    assert model.case["bus"][4,2] == complex(4,3)

def test_player_cross_table(model,workdir):
    """`TABLE:OBJECTNAME.PROPERTYNAME` targets an object of another table"""
    columns = ["RATE_A","gen:2.PMAX","bus:4.PD.real","bus:x:y.QD"]
    file = write_player(workdir/"p.csv",columns,[[11,22,33,44],[0,0,0,0]])
    PPData(model).set_player("branch",file,parent=1,mappings={"bus":{"x:y":5,"4":4}})
    assert PPSolver(model).update_players(T0) == 0
    assert rate(model,1) == 11
    assert model.get_data("gen").loc[2,"PMAX"] == 22
    assert list(model.get_data("bus").loc[[4,5],["PD","QD"]].values.diagonal()) == [33,44]

def test_player_cross_table_mapping(model,workdir):
    """`mappings` resolves object names of other tables; `nosuch:` is part of the name"""
    file = write_player(workdir/"p.csv",["gen:3.PMAX","bus:9.PD"],[[5,6],[0,0]])
    PPData(model).set_player("branch",file,mappings={"gen":("GEN_BUS",int),"bus":"BUS_I"})
    assert PPSolver(model).update_players(T0) == 0
    assert model.get_data("gen").set_index("GEN_BUS").loc[3,"PMAX"] == 5
    assert model.get_data("bus").set_index("BUS_I").loc[9,"PD"] == 6
    file = write_player(workdir/"q.csv",["nosuch:1.RATE_A"],[[7],[0]])
    PPData(model).set_player("branch",file,mapping={"nosuch:1":2})
    assert PPSolver(model).update_players(T0) == 0
    assert rate(model,2) == 7

@pytest.mark.parametrize("columns,match",[
    (["gen:PMAX"],"must name an object"),
    (["gen:.PMAX"],"must name an object"),
    (["gen:1.RATE_A"],"not found in gen"),
    (["gen:9.PMAX"],"not in gen"),
    (["RATE_A"],"requires a parent"),
    (["RATE_A.real"],"ambiguous"),
    (["NOSUCH"],"not found"),
    (["99.RATE_A"],"not in branch"),
    (["x.RATE_A"],"not a branch row number"),
    ([".RATE_A.whole"],"part"),
    ])
def test_player_errors(model,workdir,columns,match):
    """Invalid column names are rejected when the player is set"""
    file = write_player(workdir/"p.csv",columns,[[1],[2]])
    with pytest.raises(AssertionError,match=match):
        PPData(model).set_player("branch",file)
    assert not model.players

def test_player_missing_time(model,workdir):
    """A timestep missing from a player is counted as an error"""
    file = write_player(workdir/"p.csv",["RATE_A"],[[1],[2]])
    PPData(model).set_player("branch",file,parent=0)
    with pytest.warns(UserWarning):
        assert PPSolver(model).update_players(T1 + pd.Timedelta("1h")) == 1

def test_player_timeseries_and_json(model,workdir):
    """Players apply in `run_timeseries` and survive `save()`/`load()`"""
    data = PPData(model)
    data.set_input("bus","PD",f"{DATADIR}/load.csv")
    file = write_player(workdir/"p.csv",["PMAX","1.PMAX"],[[250,270],[260,280]])
    data.set_player("gen",file,parent=0)
    start = dt.datetime(2020,7,31,17,0,0,0,pytz.UTC)
    end = dt.datetime(2020,7,31,18,0,0,0,pytz.UTC)
    assert PPSolver(model).run_timeseries(start=start,end=end,freq="1h") is None, model.errors
    assert list(model.get_data("gen").loc[[0,1],"PMAX"]) == [260,280]

    model.save("model.json")
    loaded = PPModel()
    loaded.load("model.json")
    assert list(loaded.players) == [file]
    assert loaded.players[file]["targets"] == model.players[file]["targets"]
    assert PPSolver(loaded).update_players(T0) == 0
    assert np.allclose(loaded.get_data("gen").loc[[0,1],"PMAX"],[250,270])
