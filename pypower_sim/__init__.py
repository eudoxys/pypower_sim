"""PyPOWER timeseries simulation

The `pypower_sim` package support time-series simulation using 
[pypower](https://github.com/rwl/pypower). The simulation accept
`pypower` case files for the model and CSV files for time-series
data. The output can be CSV, PNG, or Google Earth KML files, as
shown below in Figure 1.

```mermaid
flowchart LR

    PPCLI --> pypower_sim
    
    case(case) --> PPModel
    inputs(csv) --> input[PPData]

    subgraph pypower_sim
        input --> PPSolver
        PPModel --> PPSolver
        PPSolver --> PPModel
        PPModel --> PPPlots
        PPModel --> kml
    end

    PPSolver --> data(csv)
    PPPlots --> images(png)
    kml --> viewer[Google Earth]
```
Figure 1: `pypower_sim` package architecture

# Description

The `pypower_sim.ppmodel.PPModel` class manages the `case` data structures
needed by `pypower` solvers. Although users can access `case` data directly,
convenience methods such as `pypower_sim.ppmodel.PPModel.get_data` can be
used to convert the `case` data to Pandas data frames where the columns are
labeled according the `pypower` specifications.

In addition, several items have been added to the `case` data structures to
enable geographic data models, distributed generation modeling, and load
modeling extensions. See `pypower_sim.ppgis.PPGIS`,
`pypower_sim.ppdgen.PPDGen`, and `pypower_sim.ppload.PPLoad` for details.

# Installation

    python3 -m venv .venv
    . .venv/bin/activate
    pip install git+https://github.com/eudoxys.com/pypower_sim

# Examples

Examples can be accessed by cloning the source code. They are not included
in normal installations.

## Preparation

Close the `pypower_sim` repository from GitHub

    git clone https://github.com/eudoxys/pypower_sim
    pip install ./pypower_sim
    cd ./pypower_sim/examples
    python3

Alternatively, you can download the files from https://github.com/eudoxys/pypower_sim/tree/main/examples.

## Load a case

    from pypower_sim import PPModel
    from wecc240 import wecc240

    model = PPModel(case=wecc240)

## Run one-shot solvers

    from pypower_sim import PPSolver

    solver=PPSolver(model)
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

    import datetime as tz
    import pytz

    start = dt.datetime(2020,7,31,17,0,0,0,pytz.UTC)
    end = dt.datetime(2020,8,1,16,0,0,0,pytz.UTC)
    solver.run_timeseries(
        start=start,
        end=end,
        freq="1h",
        progress=print,
        )

# See also

- [Eudoxys PyPOWER Distribution](https://github.com/eudoxys/pypower)

# Package information

- Source code: https://github.com/eudoxys/pypower_sim

- Documentation: https://www.eudoxys.com/pypower_sim

- Issues: https://github.com/eudoxys/pypower_sim/issues

- License: https://github.com/eudoxys/pypower_sim/blob/main/LICENSE

- Dependencies:

    - [matplotlib](https://pypi.org/project/matplotlib/)
    - [openpyxl](https://pypi.org/project/openpyxl/)
    - [pandas](https://pypi.org/project/pandas/)
    - [pytz](https://pypi.org/project/pytz/)
    - [scipy](https://pypi.org/project/scipy/)
    - [geohash](https://github.com/eudoxys/geohash)
    - [pypower](https://github.com/eudoxys/pypower)

----
"""

import sys

from .kml import KML
from .ppdata import PPData
from .ppgen import PPGen
from .ppmodel import PPModel
from .ppplots import PPPlots
from .ppsolver import PPSolver
from .ppcli import PPCLI
from .ppjson import PypowerModelDecoder, PypowerModelEncoder
from .ppgis import PPGIS

def main(*args,**kwargs):
    """Main command line processor

    # Arguments

    - `*args`: command line arguments (see `PPCLI`)

    - `**kwargs`: command line processor options (see `PPCLI`)

    # Returns

    - `int`: exit code (see `PPCLI`)
    """

    cli = PPCLI(*args,**kwargs)

    return cli.exitcode

if __name__ == "__main__":

    sys.exit(main())
