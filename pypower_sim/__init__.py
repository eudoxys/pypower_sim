"""PyPOWER timeseries simulation

Syntax
------

    ppsim [-h] [-d] [-w] [--version] [-C CASE] FILENAME print

Positional arguments
--------------------

  - `FILENAME`: `pypower` case file (a Python file defining a case function)

  - `print`: print the case's bus, branch, generator, and cost tables

Optional arguments
------------------

  - `-C|--case CASE`: case function name in `FILENAME` (default is the file
    basename)

  - `-d|--debug`: enable debug logging and traceback on exceptions

  - `-h|--help`: show this help message and exit

  - `--version`: show the package version and exit

  - `-w|--warning`: enable warning messages from python

Description
-----------

The `pypower_sim` package supports time-series simulation and optimizations
using[`pypower`](https://github.com/rwl/pypower) and [`cvxpy`]
(https://www.cvxpy.org/). The simulation accept `pypower` case files for the
model and CSV files for time-series data. The output can be CSV, PNG, or
Google Earth KML files, as shown below in Figure 1.

```mermaid
flowchart LR

    CLI[<a href="pypower_sim/cli.html" target="_blank">ppsim</a>] --> pypower_sim
    
    case[\\<a href="https://rwl.github.io/PYPOWER/api/pypower.caseformat-module.html" target="_blank">case</a>\\] --> PPModel[<a href="pypower_sim/ppmodel.html" target="_blank">PPModel</a>]
    inputs[\\csv\\] --> input[<a href="pypower_sim/ppdata.html" target="_blank">PPData</a>]

    subgraph pypower_sim
        input --> PPSolver[<a href="pypower_sim/ppsolver.html" target="_blank">PPSolver</a>]
        PPModel <--> PPSolver
        PPModel -----> PPPlots[<a href="pypower_sim/ppplots.html" target="_blank">PPPlots</a>]
        runoce <--> PPGraph[<a href="pypower_sim/ppgraph.html" target="_blank">PPGraph</a>]
        PPModel -----> KML[<a href="pypower_sim/kml.html" target="_blank">KML</a>]
        PPSolver <---> runosp[runoce]
        PPSolver ----> outputs[<a href="pypower_sim/ppdata.html" target="_blank">PPData</a>]
    end

    runoce <---> cvxpy[<a href="https://www.cvxpy.org/" target="_blank">cvxpy</a>]
    subgraph pypower
        runpf[<a href="https://rwl.github.io/PYPOWER/api/pypower.runpf-module.html" target="_blank">runpf</a>]
        runopf[<a href="https://rwl.github.io/PYPOWER/api/pypower.runopf-module.html" target="_blank">runopf</a>]
    end

    PPSolver <-----> pypower
    
    outputs --> data[/csv/]
    PPPlots --> images[/png/]
    KML --> kml[/kml/]
```
Figure 1: `pypower_sim` package architecture

Optimal DC and AC powerflow and optimal generator, capacitor, and condenser
sizing/placement solvers are available for `pypower` cases. The optimizers
can be used to ensure that powerflow solutions are feasible and do not
require load curtailment.

Model data
----------

The `pypower_sim.ppmodel.PPModel` class manages the `case` data structures
needed by `pypower` solvers. Although users can access `case` data directly,
convenience methods such as `pypower_sim.ppmodel.PPModel.get_data` can be
used to convert the `case` data to Pandas data frames where the columns are
labeled according the `pypower` specifications.

In addition, a `gis` item can be added to the `case` data structure to
enable geographic data models. See `pypower_sim.ppgis.PPGIS` for details.

What is not promised
--------------------

  - The command line implements only `print`; solving, plotting, and
    time-series simulation are Python-only.

  - The optimal capacity expansion solver (`PPSolver.solve_oce`) is
    experimental: it currently returns `False` on all bundled test cases,
    even when it proposes capacity updates.

  - AC OPF does not converge for every case (e.g., `case240_2018m`), and
    powerflow fails on `case4r` after its OPF.

  - Distributed generation and load modeling extensions are planned but not
    implemented (`pypower_sim.ppdgen` is a placeholder).

  - JSON case files (`PPModel(case="file.json")`) are not supported; use
    `PPModel.load()` for models saved with `PPModel.save()`.

Installation
------------

Install using Eudoxys Package Manager:

    pip install --upgrade eudoxys
    epm install pypower_sim

Install using `pip`:

    pip install git+https://github.com/eudoxys/pypower_sim

Examples
--------

[Examples](https://github.com/eudoxys/pypower_sim/tree/main/examples) can be accessed by cloning the source code. They are not included
in normal installations.

### Preparation

Clone the `pypower_sim` repository from GitHub

    git clone https://github.com/eudoxys/pypower_sim
    pip install ./pypower_sim
    cd ./pypower_sim/examples
    python3

Alternatively, you can download the files from https://github.com/eudoxys/pypower_sim/tree/main/examples.

### Load a case

    from pypower_sim import PPModel
    from wecc240 import wecc240

    model = PPModel(case=wecc240)

### Run one-shot solvers

    from pypower_sim import PPSolver

    solver=PPSolver(model)
    solver.solve_opf()
    solver.solve_opf(use_acopf=True)
    solver.solve_pf(with_result=True)


### Plot bus voltages

    from pypower_sim import PPPlots

    plotter = PPPlots(model)
    plotter.voltage().savefig("wecc240_voltage.png")

### Load time-series inputs, outputs, and recorders

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

### Run time-series solution

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

See also
--------

- [Eudoxys PyPOWER Distribution](https://github.com/eudoxys/pypower)

Package information
-------------------

- Source code: https://github.com/eudoxys/pypower_sim

- Documentation: https://www.eudoxys.com/pypower_sim

- Issues: https://github.com/eudoxys/pypower_sim/issues

- License: https://github.com/eudoxys/pypower_sim/blob/main/LICENSE

- Dependencies:

    - [cvxpy](https://pypi.org/project/cvxpy)
    - [matplotlib](https://pypi.org/project/matplotlib/)
    - [numpy](https://pypi.org/project/numpy/)
    - [openpyxl](https://pypi.org/project/openpyxl/)
    - [pandas](https://pypi.org/project/pandas/)
    - [pytz](https://pypi.org/project/pytz/)
    - [scipy](https://pypi.org/project/scipy/)
    - [geohash](https://github.com/eudoxys/geohash)
    - [pypower](https://github.com/eudoxys/pypower)

----
"""

try:
    import importlib.metadata
    __version__ = importlib.metadata.version("pypower_sim")
except Exception: # pylint: disable=broad-exception-caught
    __version__ = "dev"

from .kml import KML
from .ppdata import PPData
from .ppgen import PPGen
from .ppmodel import PPModel
from .ppplots import PPPlots
from .ppsolver import PPSolver
from .ppcli import PPCLI
from .ppjson import PypowerModelDecoder, PypowerModelEncoder
from .ppgis import PPGIS
from .ppgraph import PPGraph

from ._fuzzy import Fuzzy
from ._runoce import runoce, OceOptions
from ._violations import violations

from .cli import main
