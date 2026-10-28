# Package

- [x] Eudoxys package layout, pytest suite, and CI (see `MANIFEST.md`)

- [x] `ppsim` CLI with `print` (see `pypower_sim.cli`)

- [ ] Remove deprecated `pypower_sim.ppcli.PPCLI` (separate PR)

- [ ] CLI commands `plot`, `solve`, and `simulate`

- [x] `DESIGN.md` written from the implementation (draft 1; needs maintainer
  review)

- [x] Stop tracking `docs/` (built by `make docs` and the Pages workflow)

- [ ] Resolve the open questions in `DESIGN.md` (input interpolation,
  players, OCE success criterion, distributed generation scope)

- [ ] First release (`YEAR.MONTH.0`)

# Solvers

- [ ] `PPSolver.solve_oce()` reports failure on all bundled cases

- [ ] AC OPF on `case240_2018m`; powerflow on `case4r` after OPF

- [ ] Fix `PPModel.set_case("call@file.py")`

- [ ] Fix `PPData.set_player()` and the inverted initial-OCE error checks in
  `run_timeseries`

# Data

- [ ] Provenance (source URL, retrieval date) for `_test/data/` cases and
  the `examples/` data (see the TODO in `examples/README.md`)

- [ ] Remove unused `marimo` and `openpyxl` requirements, if confirmed unused
