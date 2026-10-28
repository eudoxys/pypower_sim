"""Tests of `pypower_sim.ppmodel.PPModel` (fast)"""

import pytest
import pandas as pd

from pypower_sim import PPModel

from conftest import SMALL_CASES, case_file

@pytest.mark.parametrize("name",SMALL_CASES)
def test_load_case_file(name):
    """A case file loads and its tables match the case arrays"""
    model = PPModel(case=case_file(name))
    assert model.name == name
    for table in ["bus","branch","gen"]:
        data = model.get_data(table)
        assert isinstance(data,pd.DataFrame)
        assert len(data) == len(model.case[table])
        # result columns (e.g. LAM_P) are present only after a solve
        header = PPModel.get_header(table)
        assert list(data.columns) == header[:len(data.columns)]

def test_load_case_callable():
    """A case function gives the same model as its file"""
    from_file = PPModel(case=case_file("case9"))
    module = {}
    with open(case_file("case9"),encoding="utf-8") as fh:
        exec(fh.read(),module) # pylint: disable=exec-used
    from_call = PPModel(name="case9",case=module["case9"])
    for table in ["bus","branch","gen","gencost"]:
        assert (from_file.case[table] == from_call.case[table]).all()

def test_default_model():
    """A model with no case has an empty but valid case structure"""
    model = PPModel()
    assert isinstance(model.case,dict)
    assert "baseMVA" in model.case

def test_mvabase_override():
    """The `mvabase` argument overrides the case base"""
    model = PPModel(case=case_file("case9"),mvabase=200.0)
    assert model.case["baseMVA"] == 200.0

def test_json_unsupported():
    """JSON case files are rejected (use `load()`)"""
    with pytest.raises(NotImplementedError):
        PPModel(case="model.json")

def test_save_load_roundtrip(workdir):
    """`save()` then `load()` preserves the case data"""
    model = PPModel(case=case_file("case14"))
    model.save("case14.json",indent=1)
    assert (workdir / "case14.json").exists()
    loaded = PPModel()
    loaded.load("case14.json")
    for table in ["bus","branch","gen","gencost"]:
        assert (loaded.get_data(table).values == model.get_data(table).values).all()
