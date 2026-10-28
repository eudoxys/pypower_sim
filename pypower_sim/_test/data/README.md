Test data
---------

Curated inputs for the tests in `pypower_sim/_test/` and for the module
`__main__` maintenance blocks (`python3 ppmodel.py` etc., run from the
`pypower_sim/` folder). Source URLs and retrieval dates were not recorded
when these files were added; fill them in when known.

Files
-----

  - `case4gsm.py`, `case4m.py`, `case4r.py`, `case6ww.py`, `case9.py`,
    `case9Q.py`, `case14.py`, `case30.py`, `case30Q.py`, `case30pwl.py`,
    `case39.py`, `case57.py`, `case118.py`, `case300.py`: standard
    MATPOWER/PYPOWER test cases (PSERC, BSD license; see each file's header).
    `case4m.py`, `case4r.py` and the `__main__` blocks that call
    `check_case.py` are local additions.

  - `case240_2011m.py`, `case240_2018m.py`, `case240_2020m.py`: WECC
    240-bus reduced models for the 2011 (Price et al.), 2018 and 2020 (Yuan
    et al.) systems, as `pypower` version 2 cases.

  - `case240_gis.csv`: bus locations and attributes for the WECC 240-bus
    cases. Columns: `BUS_I` (bus id), `LAT`, `LON` (degrees), `GEOHASH`,
    `NAME`, `GEN`, `LOAD` (pu), `BA` (balancing authority), `COUNTY` (FIPS).

  - `gis.csv`: an earlier version of the WECC 240-bus locations (`BUS_I`,
    `LAT`, `LON`, `GEOHASH`, `NAME`, `GEN`, `LOAD`), not used by the tests.

  - `load.csv`: synthetic hourly load, 2020-07-31 17:00 to 2020-08-01
    17:00 UTC (hour beginning). Columns are bus row numbers (`1`, `2`, `3`,
    `6`); a synthetic profile that steps up then down. Used with `PPData.set_input` and a
    `scale`.

  - `check_case.py`: helper the standard case files call from their
    `__main__` blocks (`python3 case9.py`) to run PF, OPF and OCE on a case.
