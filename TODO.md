# Package

- [x] Eudoxys package layout, pytest suite, and CI (see `MANIFEST.md`)

- [x] `ppsim` CLI with `print` (see `pypower_sim.cli`)

- [ ] Remove deprecated `pypower_sim.ppcli.PPCLI` (separate PR)

- [ ] CLI commands `plot`, `solve`, and `simulate`

- [ ] Write `DESIGN.md` before building distributed generation (`ppdgen.py`)
  and load modeling

- [ ] Decide whether to stop tracking `docs/` (rebuilt by the Pages workflow)

- [ ] First release (`YEAR.MONTH.0`)

# Solvers

- [ ] `PPSolver.solve_oce()` reports failure on all bundled cases

- [ ] AC OPF on `case240_2018m`; powerflow on `case4r` after OPF

- [ ] Fix `PPModel.set_case("call@file.py")`

# Data

- [ ] Provenance (source URL, retrieval date) for `_test/data/` cases and
  the `examples/` CSV files

- [ ] Remove unused `marimo` and `openpyxl` requirements, if confirmed unused
