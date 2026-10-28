# CLAUDE.md

Guidance for Claude when working in the `eudoxys/pypower_sim` repository.
General Eudoxys package conventions are in the `eudoxys-package` skill
(`eudoxys/claude`); this file records only what is specific to this package.

## What this is

A quasi-steady time-series simulator for `pypower` cases: `PPModel` holds a
`pypower` case dict, `PPData` maps CSV time series onto case columns,
`PPSolver` runs powerflow, OPF, optimal capacity expansion (OCE, via `cvxpy`)
and time-series loops, and `PPPlots`/`KML` render outputs. Python 3.12+,
alpha maturity, version `0.0.0` (unreleased).

- Docs site: https://www.eudoxys.com/pypower_sim (pdoc from docstrings)
- Maintainer: David P. Chassin
- CLI entry point: `ppsim` -> `pypower_sim.cli:main` (only `print` works)

## Where things are

- `MANIFEST.md`: map of every file, a task index, and what not to read.
  Read it first instead of listing folders.
- `DESIGN.md`: the design as implemented, numbered decisions, known issues,
  and open questions. It was written from the code (draft 1) and awaits
  maintainer review; follow its decisions and ask before departing from one.
  Update it before adding a major subsystem (e.g. distributed generation in
  `ppdgen.py`).

Sibling Eudoxys dependencies:
- `pypower` (`eudoxys/pypower`): the solvers (`runpf`, `runopf`) and `idx_*`
  column indexes.
- `geohash` (`eudoxys/geohash`): bus location hashing in `ppgis`/`ppgen`.

## Core conventions

- Case data is a `pypower` version 2 dict (`version`, `baseMVA`, `bus`,
  `branch`, `gen`, `gencost`, `dcline`, `dclinecost`, plus the local `gis`
  extension). Arrays are numpy; column positions come from `idx_*` classes
  (`pypower.idx_*`, plus `idx_gis`, `idx_dcline`, `idx_construction` here).
- `PPModel.get_data(name)` returns the array as a DataFrame with `pypower`
  column names; result columns (`LAM_P`, `MU_*`, ...) exist only after a solve.
- Case files are Python modules defining a function with the file's
  basename that returns the case dict (`case9.py` -> `case9()`).
- Solver methods return `True`/`False` (optionally `(ok, result)` with
  `with_result=True`) and record problems in `PPModel.errors`; they don't
  raise on non-convergence.
- `PPData` input CSVs are indexed by timestamp; columns are case row
  numbers unless a `mapping` is given. Timestamps are UTC.

## Style

- Most docstrings use markdown headings (`# Arguments`, `# Returns`) rather
  than the dashed-underline sections the skill prefers. New modules
  (`cli.py`) use dashed underlines; don't mass-convert existing docstrings.
- `__main__` blocks in modules are maintenance scripts run from the
  `pypower_sim/` folder with bare imports (`from ppmodel import PPModel`),
  over the cases in `_test/data/`.

## Build, test, docs

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip && pip install -e . pytest
pytest                  # ~20 s; pytest -m "not slow" skips WECC 240-bus
make docs
python .github/check_manifest.py
```

No network or credentials are needed. CI (`validate.yaml`) installs
non-editable, checks the manifest, runs pylint (informational) and pytest.

## Known rough edges (don't "fix" silently; mention or ask)

- `PPSolver.solve_oce()` returns `False` on every bundled case, even when it
  proposes capacity updates (`test_oce` is `xfail`).
- AC OPF does not converge on `case240_2018m` (`xfail`); powerflow fails on
  `case4r` after OPF; `python3 ppsolver.py` reports OCE errors on all cases.
- `PPModel.set_case("call@file.py")` is documented but broken: it loads the
  whole string as the path. The CLI's `-C` loads the module itself instead.
- `PPModel.print()` binds `file=sys.stdout` at import time and writes blank
  lines to stderr; pass `file=` explicitly when capturing output.
- `ppcli.PPCLI` is deprecated and non-functional (it discards
  `parse_args()`); it is still exported. Remove it in its own PR.
- `ppdgen.py` is entirely commented out.
- `marimo` and `openpyxl` are in `requirements.txt` but not imported.
- `kml.py`'s `__main__` block calls macOS `open`.
- `examples/ca-*.csv` (~150 MB, ResStock-style end-use profiles),
  `examples/wecc_eia861.csv`, and `examples/wecc240.py` have no recorded
  source; see the TODO in `examples/README.md`.
- `PPData.set_player()` is broken and players are never usable; the initial
  OCE branch of `run_timeseries` records errors on success (see `DESIGN.md`).
- pandas emits `inplace` deprecation warnings from `ppgis.py`.

## Working rules

- Don't commit generated output (including `docs/`): test output goes to pytest temp folders;
  `.kml` files and plots from maintenance runs are gitignored.
- `_test/data/` and `examples/` data are curated; don't regenerate or
  reformat them. Don't read the large `examples/*.csv`; use `head`.
- When a public signature or output changes, update the module docstring,
  the package docstring (`__init__.py`), `README.md`, and `DESIGN.md` in the
  same PR.
- Update `MANIFEST.md` when adding, moving, or deleting files.
- Conventional small test case: `_test/data/case9.py`.
