Testing
-------

From the repository root:

	python3 -m venv .venv
	. .venv/bin/activate
	python3 -m pip install --upgrade pip
	pip install -e . pytest
	pytest

The tests need no network access or credentials, download nothing, and
write output only to pytest temporary folders. The full suite takes about
20 seconds; skip the WECC 240-bus cases with `pytest -m "not slow"`.

Two tests are expected to fail (`xfail`) until known solver issues are
resolved (see `CLAUDE.md`, "Known rough edges"):

  - `test_ppsolver.py::test_oce`: `PPSolver.solve_oce()` returns `False`;
  - `test_wecc240.py::test_acopf[case240_2018m]`: AC OPF does not converge.

Test inputs are in [`data/`](data/README.md).

`conftest.py`
-------------

Defines `DATADIR`, the case lists `SMALL_CASES` and `WECC_CASES`,
`case_file(name)`, and the `workdir` fixture that runs a test in a
temporary folder.

`test_ppmodel.py`
-----------------

Fast. Loads each small case from file and from its case function, checks
table shapes and column headers, the `mvabase` override, rejection of JSON
case files, and the `save()`/`load()` JSON round trip.

	pytest pypower_sim/_test/test_ppmodel.py

`test_ppsolver.py`
------------------

Fast. Powerflow and AC OPF on each small case (no violations after OPF),
the power balance on `case9`, and a 7-hour time-series run on `case9` using
`data/load.csv` (one output row per timestep per bus, voltages in range,
cost rising with load).

	pytest pypower_sim/_test/test_ppsolver.py

`test_cli.py`
-------------

Fast. The `ppsim` command line: `print`, `-C`, `--version`, and the exit
codes for missing files, unknown cases, and unimplemented commands.

	pytest pypower_sim/_test/test_cli.py

`test_wecc240.py`
-----------------

Slow (marked `slow`, about 10 seconds). Powerflow and AC OPF on the three
WECC 240-bus cases and attaching `data/case240_gis.csv` with `PPGIS`.

	pytest -m slow
