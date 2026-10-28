# eudoxys/pypower_sim — package design (draft 1)

Status: draft written from the existing implementation, 2026-10-28; not yet
reviewed with the maintainer. It records what the code does now and the
decisions that behavior implies. Where the code is incomplete or inconsistent,
the gap is listed under Known issues or Open questions, not settled as a
decision.

## Purpose

`pypower_sim` runs quasi-steady time-series (QSTS) simulations of
transmission networks expressed as `pypower` cases. It wraps the `pypower`
powerflow (PF) and optimal powerflow (OPF) solvers with:

- a model object that holds and converts case data (`PPModel`);
- time-series inputs and outputs read from and written to CSV (`PPData`);
- a solver driver for single solves, an experimental convex optimal capacity
  expansion (OCE) solver, and the time-series loop (`PPSolver`);
- geographic (GIS) bus data, KML export, plots, and network graph analysis.

Excluded: the PF/OPF numerical solvers themselves (`eudoxys/pypower`),
geohash encoding (`eudoxys/geohash`), dynamic/transient simulation,
distribution networks, and data acquisition (callers supply case files and
CSVs; `PPGen` subclasses supply generator fleet data).

Coverage: any `pypower` version 2 case. The bundled cases are the standard
MATPOWER/PYPOWER test cases and WECC 240-bus models (2011, 2018, 2020).

## Decisions settled

These are inferred from the implementation (2026-10-28) and need the
maintainer's confirmation.

| # | Decision |
|---|---|
| 1 | The model is the `pypower` version 2 case dict (`version`, `baseMVA`, `bus`, `branch`, `gen`, optional `gencost`, `dcline`, `dclinecost`), held unchanged in `PPModel.case` so it can be passed straight to `pypower` solvers. |
| 2 | Extensions to the case (`gis`; `cost` after OPF) are extra keys in the same dict, not a parallel structure. |
| 3 | Column positions come from `idx_*` classes (`pypower.idx_*` plus local `idx_gis`, `idx_dcline`, `idx_construction`); `PPModel.get_header()` derives the column names from them, and `get_data()` returns DataFrames with those names and the integer dtypes in `PPModel.types_idx`. |
| 4 | Case files are Python modules defining a function named after the file that returns the case dict; `PPModel(case=...)` also accepts the function or the dict. |
| 5 | Models persist as JSON (`PPModel.save()`/`load()`, via `ppjson`), including inputs, outputs, recorders, and options, and case files can be written back as Python (`save_case()`). |
| 6 | Solvers do not raise on non-convergence: `solve_*` return `True`/`False` (or `(ok, result)` with `with_result=True`) and `run_timeseries` returns `None` or the error list; messages go to `PPModel.errors`. |
| 7 | Solvers update `PPModel.case` in place, controlled by `update` (`always`, `success` (default), `failure`, `never`). |
| 8 | Each time step solves OPF (AC by default, DC with `use_acopf=False`, skipped with `None` or when there is no `gencost`), then PF from the OPF result; outputs are written after PF. |
| 9 | Time steps come from `pandas.date_range(*args,**kwargs)` and are converted to UTC; input rows are looked up by exact timestamp (no interpolation). |
| 10 | Inputs and outputs are CSV files with a timestamp index and one column per case row (row number, not bus id) unless a `mapping` is given; values are `data * scale + offset`. |
| 11 | Output files are opened at the start of a run and written one row per completed step, so partial runs leave usable files. |
| 12 | OCE is a convex relaxation of OPF solved with `cvxpy` (method from `eudoxys/fast_oce`); it adds generation and dynamic var capacity only at non-PQ buses and static `BS` at PQ buses, and never adds branches. |
| 13 | The CLI (`ppsim`) is a thin layer over `PPModel`; anything it does is also available from Python. |

## Conventions inherited

From the `eudoxys-package` skill and sibling packages:

- packaging, CI, docs, and release per `eudoxys-package`;
- `pypower` case format, `ppoption` option names, and `idx_*` indexes from
  `eudoxys/pypower`;
- geohash bus locations from `eudoxys/geohash`.

## Module layout

```
pypower_sim/
  # model
  ppmodel.py      PPModel(name=None,version=None,mvabase=None,case=None): case data, idx_* tables, save/load, print, violations
  ppjson.py       PypowerModelEncoder, PypowerModelDecoder: JSON for numpy arrays and model dicts
  ppgis.py        PPGIS(model,data,columns=None,update=None,**read_csv): bus locations into case["gis"]; idx_gis
  # solving
  ppsolver.py     PPSolver(model): solve_pf, solve_opf, solve_oce, run_timeseries
  _runoce.py      runoce(data,options=None,**kwargs), OceOptions: cvxpy OCE problem
  _violations.py  violations(data,precision,error): limit violations in a solved case
  _fuzzy.py       Fuzzy: float with tolerance comparisons
  # data I/O
  ppdata.py       PPData(model): set_input, set_player, set_output, set_recorder
  ppgen.py        PPGen(source,cache): abstract generator-fleet source -> gen/gencost rows
  # outputs and analysis
  ppplots.py      PPPlots(model): voltage, generation, load profiles (matplotlib)
  kml.py          KML(kmlfile,name): Google Earth styles, markers, lines
  ppgraph.py      PPGraph(model): degree, adjacency, incidence, Laplacian, spectral analysis
  # interface
  cli.py          main(argv): ppsim FILENAME print
  ppcli.py        PPCLI: deprecated
  # planned
  ppdgen.py       distributed generation (placeholder, commented out)
                  load modeling extensions
```

### Contracts

Every helper takes the model as its first argument and works on
`model.case` in place:

```python
PPSolver(model)  PPData(model)  PPPlots(model)  PPGraph(model)  PPGIS(model,data)
```

| class | reads | writes |
|---|---|---|
| `PPModel` | case file/function/dict, JSON | `case`, `inputs`, `outputs`, `recorders`, `options`, `errors`, `profile` |
| `PPData` | input CSVs | `model.inputs`, `model.outputs`, `model.recorders` |
| `PPSolver` | `model.case`, `model.options`, inputs | `model.case` (per `update`), output/recorder CSVs, `model.errors`, `model.profile` |
| `PPGIS` | GIS CSV/DataFrame | `model.case["gis"]` |
| `PPPlots` | `model.case` | matplotlib figures |
| `PPGraph` | `model.case` (`bus`, `branch`) | sparse matrices, spectral results |

## Data sources and method

| output | primary source | fallback | anchor/reconciliation |
|---|---|---|---|
| network, limits, costs | case file | `PPModel.default_model` (empty case) | none |
| time-varying case values | `PPData` input CSVs | case values unchanged | none |
| bus locations | GIS CSV via `PPGIS` | none (no `gis` key) | none |
| generator fleet | `PPGen` subclass data | case `gen`/`gencost` | none |

`PPSolver.run_timeseries(*args,**kwargs)`:

1. **Check** callbacks and options; clear `model.errors`; build the time
   range with `pandas.date_range`.
2. **Optional initial OCE** (`with_oce=OceOptions(...)`): find the peak of the
   single `bus.PD` input, apply inputs at that time, and if OPF or PF fails
   run OCE to add capacity.
3. **Open** recorder and output files and write headers.
4. **For each step (UTC):** apply inputs (`data.loc[t] * scale` into the
   mapped case rows); solve OPF if enabled and `gencost` exists; solve PF;
   write outputs and recorders; optionally check violations; call
   `progress`, `call_on_fail`, `stop_test` callbacks. With
   `stop_on_fail=True` (default) the first failure ends the run.
5. **Profile:** record iterations and OPF/PF/total times in `model.profile`.

Failures are reported (in `model.errors` and callbacks), never forced to
succeed.

## Outputs

### Output CSV (`PPData.set_output`)

Key: `timestamp`.

| column | dtype | notes |
|---|---|---|
| `timestamp` | str | `%Y-%m-%d %H:%M:%S %Z`, UTC |
| `0`, `1`, ... | float | case row number (default mapping); value of the chosen case column × `scale` + `offset`, formatted with `formatting`; units are the `pypower` units of that column (MW, MVAr, p.u., degrees) |

Example rows (from `examples/wecc240_bus_vm.csv`):

```
timestamp,0,1,2,...
2020-07-31 17:00:00 UTC,0.986,0.969,0.969,...
```

### Recorder CSV (`PPData.set_recorder`)

Key: `timestamp`; one column per recorded target (e.g. `cost`, scaled by
`baseMVA` in the examples).

### `PPModel.profile`

`Expected iterations`, `Completed iterations`, `Total OPF time (s)`,
`Fraction OPF time (s/s)`, `Total PF time (s)`, `Fraction PF time (s/s)`,
`Total run time (s)`, `Iteration time (s/iter)`.

## Files and formats

- Case files: Python (`case<name>.py`, function `case<name>()`), as written
  by `PPModel.save_case()`.
- Saved models: JSON (`PPModel.save()`), with `application`, `version`
  (package version), `name`, `case`, `inputs`, `outputs`, `recorders`,
  `options`.
- Time series: CSV, timestamp first column; input timestamps must match the
  simulation steps exactly.
- Maps: KML (`PPModel.save_kml()`, `PPGen.to_kml()`); plots: any matplotlib
  format.

## Dependencies and gaps in sibling packages

| need | status |
|---|---|
| PF, AC/DC OPF solvers, `ppoption` | `pypower.runpf`, `runacopf`, `rundcopf` ✔ |
| `idx_dcline` column constants | none in `pypower`; local `ppmodel.idx_dcline` |
| bus geohash, nearest node | `geohash` ✔ |
| convex OCE formulation | local `_runoce.py` (from `eudoxys/fast_oce`) |

## Known issues to handle

- `solve_oce()` returns `False` on every bundled case even when it proposes
  updates; its success criterion needs review.
- AC OPF does not converge on `case240_2018m`; PF fails on `case4r` after
  OPF.
- `PPData.set_player()` fails before storing anything (`columns.values` is
  a method, and `mapping` is undefined), so players are not usable although
  the docs and the `run_timeseries` diagram describe them.
- In `run_timeseries`'s initial OCE, `if self.solve_oce(...)` records
  "initial OCE failed" on success, and "initial OPF/PF failed" is recorded
  when OPF/PF succeed (both conditions appear inverted).
- Input lookup by exact timestamp: a step missing from an input file is an
  input error, not an interpolation.
- `PPModel.set_case("call@file.py")` does not work as documented.
- OCE ignores `PMIN > 0` (treated as zero; reported as a warning).
- Assumption: a bus `BASE_KV` of zero is replaced by 230 kV
  (`PPModel.default_values`); override by editing that class attribute.

## Open questions

1. Should `run_timeseries` interpolate or forward-fill inputs between
   timestamps, or keep exact matching?
2. Should players be implemented or removed from `PPData`?
3. What defines OCE success: the `cvxpy` status, a subsequent feasible
   OPF/PF, or no capacity added?
4. Scope of the planned distributed generation (`ppdgen.py`) and load
   modeling extensions: data sources (e.g. the ResStock-style profiles and
   EIA-861 data in `examples/`), and whether they belong here or in a
   sibling package.
