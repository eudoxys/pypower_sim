"""Tests of the `ppsim` command line (`pypower_sim.cli`, fast)"""

from pypower_sim.cli import main, E_OK, E_SYNTAX, E_FAILED

from conftest import case_file

def test_print(capsys):
    """`print` writes the case tables and succeeds"""
    assert main([case_file("case9"),"print"]) == E_OK
    assert "BUS_I" in capsys.readouterr().out

def test_case_option(capsys):
    """`-C` selects the case function by name"""
    assert main([case_file("case9"),"print","-C","case9"]) == E_OK
    assert "BUS_I" in capsys.readouterr().out

def test_missing_file(capsys):
    """A missing file is a one-line error and a failure exit code"""
    assert main(["nosuchfile.py","print"]) == E_FAILED
    assert capsys.readouterr().err.startswith("ERROR [ppsim]:")

def test_missing_case(capsys):
    """An unknown case name is a failure"""
    assert main([case_file("case9"),"print","-C","nosuchcase"]) == E_FAILED
    assert "nosuchcase" in capsys.readouterr().err

def test_bad_command():
    """Unimplemented commands are syntax errors"""
    assert main([case_file("case9"),"plot"]) == E_SYNTAX

def test_version(capsys):
    """`--version` prints the version and succeeds"""
    assert main(["--version"]) == E_OK
    assert capsys.readouterr().out.startswith("ppsim ")
