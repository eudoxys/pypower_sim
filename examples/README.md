To run this example do the following

# Setup the runtime environment 

## Clone the package
    
    git clone https://github.com/eudoxys/pypower_sim
    python3 -m venv .venv
    . .venv/bin/activate
    pip install ./pypower_sim

## Change to the examples folder

    cd pypower_sim/examples

## Start python interpreter

    python3

# Run the examples in Python interpreter

## Load a case

    from pypower_sim import PPModel
    from wecc240 import wecc240
    
    model = PPModel(case=wecc240)

## Run one-shot solvers

    from pypower_sim import PPSolver
    
    solver = PPSolver(model)
    solver.solve_opf()
    solver.solve_opf(use_acopf=True)
    solver.solve_pf(with_result=True)

## Plot bus voltages
    from pypower_sim import PPPlots
    
    plotter = PPPlots(model)
    plotter.voltage().savefig("wecc240_voltage.png")
    
## Load time-series inputs, outputs, and recorders

    from pypower_sim import PPData
    
    data = PPData(model)
    data.set_input("bus","PD","load.csv",scale=10)
    data.set_input("bus","QD","load.csv",scale=1)
    data.set_output("bus","VM","wecc240_bus_vm.csv",formatting=".3f")

    data.set_output("bus","VA","wecc240_bus_va.csv",formatting=".4f")
    data.set_output("bus","PD","wecc240_bus_pd.csv",formatting=".4f")
    data.set_output("bus","QD","wecc240_bus_qd.csv",formatting=".4f")

    data.set_recorder("wecc240_cost.csv","cost",["cost"],
        scale=model.case['baseMVA'],formatting=".2f")
    
## Run time-series solution

    import datetime as dt
    import pytz

    start = dt.datetime(2020,7,31,17,0,0,0,pytz.UTC)
    end = dt.datetime(2020,8,1,16,0,0,0,pytz.UTC)
    solver.run_timeseries(
        start=start,
        end=end,
        freq="1h",
        progress=print,
        )

# Data

TODO: document the provenance of each data file below (source publisher,
dataset name, URL, retrieval date), any cleaning applied, and summary
statistics so the files can be reproduced and checked (see the
`eudoxys-package` skill, data folder READMEs).

## Files

- `case9.py`: standard 9-bus MATPOWER/PYPOWER case (PSERC, BSD license).
  Used by `pftf.py`.
- `wecc240.py`: WECC 240-bus case, saved by `PPModel.save_case()` on
  2025-11-27. Source model not recorded (TODO).
- `pftf.py`: computes powerflow transfer factors (line-outage flow changes)
  for `case9.py`.
- `load.csv`: synthetic hourly load, 2020-07-31 17:00 to 2020-08-01 17:00
  UTC; columns are bus row numbers (`1`, `2`, `3`, `6`). Same as
  `pypower_sim/_test/data/load.csv`.
- `wecc240_bus_vm.csv`, `wecc240_bus_va.csv`, `wecc240_bus_pd.csv`,
  `wecc240_bus_qd.csv`, `wecc240_cost.csv`: outputs of the time-series
  example above (24 hourly rows from 2020-07-31 17:00 UTC; columns are bus
  row numbers; VM in p.u., VA in degrees, PD in MW, QD in MVAr, cost in $).
- `wecc240_voltage.png`: output of the voltage plot example above.
- `ca-mobile_home.csv`, `ca-multi-family_with_2_-_4_units.csv`,
  `ca-multi-family_with_5plus_units.csv`, `ca-single-family_attached.csv`,
  `ca-single-family_detached.csv`: California residential end-use
  electricity profiles by building type, 15-minute, 2018 (35,040 rows,
  ~30 MB each; columns `in.*` describe the building stock and
  `out.electricity.*.energy_consumption` the end uses). They look like NREL
  ResStock / End-Use Load Profiles aggregates, but the source and release
  are not recorded (TODO). Not used by the examples above; the commented-out
  `pypower_sim/ppdgen.py` reads their PV column.
- `wecc_eia861.csv`: monthly electricity sales by sector (MW and MWh) for
  the 11 WECC states, 2018–2022, all `STATUS` Final (659 rows). Looks like EIA Form 861M,
  but the source and retrieval date are not recorded (TODO).

## TODO

- [ ] Record source URL, release, and retrieval date for `wecc240.py`, the
  `ca-*.csv` files, and `wecc_eia861.csv`.
- [ ] Record any cleaning or aggregation applied, and summary totals for
  each data file.
- [ ] Add the script that regenerates each derived file, or mark it curated.
