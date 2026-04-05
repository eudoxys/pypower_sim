"""PyPower model accessor

The PyPower model accessor is used to manage the bus, branch, generator,
generator cost, DC line, and DC line cost data arrays in `pypower` cases. Use
the 'case' member to access the `pypower` case data.

The `save_case()` method is used to export a PyPower case file.

The `save_kml()` method is used to export a Google Earth KML file.

The `print()` method is used to output the case data in human readable form
using a Pandas data frame.

# Example

The following example constructs a new PyPower model and prints the case data.

    model = PPModel()
    print(model.case)

# Data Structures

- `pypower_sim.ppmodel.PPModel.case`: Provides all the PyPOWER case data
  needed to solve powerflow and optimal powerflows. See PyPOWER `idx_*` for
  details.

- `pypower_sim.ppmodel.PPModel.inputs`: Provides all the information required
  to read data from input files and update `case` data.

- `pypower_sim.ppmodel.PPModel.outputs`: Provides all the information required
  to read `case` data and update the output files.

- `pypower_sim.ppmodel.PPModel.recorders`: Provides all the information
  required to read `case` data and update the recorder files.

- `pypower_sim.ppmodel.PPModel.options`: Provides all the options used by the
  PyPOWER solvers. See PyPOWER `ppoption` for details.

- `pypower_sim.ppmodel.PPModel.errors`: Records all the error message emitted
  during a solver call.

- `pypower_sim.ppmodel.PPModel.profile`: Collects all the solver performance
  data obtained during a solver call.

- `pypower_sim.ppmodel.PPModel.cost`: OPF cost result (if any)

# See also

- `PPData`: Model data I/O manager

- `PPSolve`: Model solvers

- `PPPlot`: Model plotting tools

- `PSSE2PP`: Model converter from PSSE
"""
# pylint: disable=too-many-lines

import os
import sys
import io
from copy import copy
import json
import datetime as dt
from typing import Callable
import warnings
from importlib.util import spec_from_file_location, module_from_spec
from importlib.metadata import version as pkg_version

import numpy as np
import pandas as pd

# pylint: disable=unused-import
from pypower import idx_brch as idx_branch
from pypower import idx_gen, idx_bus
from pypower import idx_cost as idx_gencost
from pypower_sim.ppgis import idx_gis
# pylint: enable=unused-import
from pypower_sim.ppgraph import PPGraph
from pypower_sim.ppjson import PypowerModelEncoder, PypowerModelDecoder
from pypower_sim.kml import KML
from pypower_sim._fuzzy import Fuzzy

idx_dclinecost = idx_gencost

class idx_dcline:
    """@private Provide missing column index values that should be in `pypower.idx_dcline`"""

    # pylint: disable=invalid-name,too-few-public-methods

    F_BUS = 0
    T_BUS = 1
    BR_STATUS = 2
    PF = 3
    PT = 4
    QF = 5
    QT = 6
    VF = 7
    VT = 8
    PMIN = 9
    PMAX = 10
    QMINF = 11
    QMAXF = 12
    QMINT = 13
    QMAXT = 14
    LOSS0 = 15
    LOSS1 = 16
    MU_PMIN = 17
    MU_PMAX = 18
    MU_QMINF = 19
    MU_QMAXF = 20
    MU_QMINT = 21
    MU_QMAXT = 22

class idx_construction:
    """@private Provide column index values for construction costs"""
    # pylint: disable=invalid-name,too-few-public-methods
    BUS_I = 0
    GENERATOR = 1
    CONDENSER = 2
    REACTOR = 3
    CAPACITOR = 4

def _get_idx(name:str) -> list[str]:
    """Convert idx data to a header list

    # Arguments

    - `idx`: module containing index values

    # Returns

    - `list[str]`: ordered list of data array column header names
    """
    idx = globals()[f"idx_{name}"]
    mapping = {getattr(idx,x):x for x in dir(idx) if not x.startswith("_")}
    indexes = sorted(mapping)
    assert max(indexes) - min(indexes) + 1 == len(indexes), \
        "indexes are not strictly sequential"
    return [mapping[n] for n in indexes]

# pylint: disable=too-many-instance-attributes,too-many-public-methods
class PPModel:
    """`pypower_sim` model access class implementation"""
    default_model = {
        "version": 2,
        "baseMVA": 100.0,
        "bus": np.ndarray((0,len(_get_idx("bus")))),
        "branch": np.ndarray((0,len(_get_idx("branch")))),
        "gen" : np.ndarray((0,len(_get_idx("gen")))),
        # "gencost": np.ndarray((0,len(_get_idx("gencost")))),
        # "dcline": np.ndarray((0,len(_get_idx("dcline")))),
        # "dclinecost": np.ndarray((0,len(_get_idx("dclinecost")))),
        }
    """Default PyPOWER model"""

    standard_idx = { # list of idx values that are standard but not column names
        "bus": ["PQ","PV","REF","NONE"],
        "branch": [],
        "gen": [],
        "gencost": ["PW_LINEAR","POLYNOMIAL"],
        "dcline": [],
        "dclinecost": ["PW_LINEAR","POLYNOMIAL"],
        "gis": [],
        "construction": [],
    }
    """`idx_*` values that are excluded from the line of header indexes"""

    types_idx = { # table of non-float column data types
        "bus": {
            "BUS_I": int,
            "BUS_TYPE": int,
            "BUS_AREA": int,
            "ZONE": int,
        },
        "branch": {
            "F_BUS": int,
            "T_BUS": int,
            "BR_STATUS": int,
        },
        "gen": {
            "GEN_BUS": int,
            "GEN_STATUS": int,
        },
        "gencost": {
            "MODEL": int,
            "NCOST": int,
        },
        "dcline": {
            "F_BUS": int,
            "T_BUS": int,
            "BR_STATUS": int,
        },
        "dclinecost": {
            "MODEL": int,
            "NCOST": int,
        },
        "gis":{
            "BUS_I": int,
            "GEOHASH": str,
            "NAME": str,
        },
        "construction":{
            "BUS_I": int,
        }
    }
    """Table of `idx_*` indexes that refer to values having non-float types"""

    default_options = {

        # general options
        "VERBOSE": 0, 
        "OUT_ALL": 0,
        "OUT_ALL_LIM": -1,
        "OUT_V_LIM": 1,

        # powerflow options
        "PF_ALG": 1,
        "PF_DC": False,
        "PF_LIN_SOLVER_NR": "",

        # continuation powerflow options
        "CPF_PARAMETERIZATION": 3,
        "CPF_STOP_AT": "NOSE",
        "CPF-STEP": 0.05,
        "CPF_ADAPT_STEP": False,
        "CPF_ERROR_TOL": 1e-3,
    }
    """Solver options"""

    default_values = {
        "BASE_KV" : 230.0
    }
    """Default for missing/zero values (`None` to leave zeros)"""

    # pylint: disable=too-many-public-methods

    def __init__(self,
        name:str=None,
        version:int=None,
        mvabase:float=None,
        case:str|dict|Callable=None,
        ):
        """Create PyPower case data

        # Arguments:

        - `name`: name of the case

        - `version`: case version number

        - `mvabase`: MVA base value

        - `case`: case data, file name, or case constructor/function
        """

        # pylint: disable=too-many-instance-attributes
        self.name : str = name
        """Name of case"""

        self.case : dict[str|int|np.array] = None
        """Case data (see https://github.com/eudoxys/pypower)"""

        self.zone_kv : list[float] = None
        """Bus zone voltages for per-unit calculations"""

        self.set_case(case)
        if mvabase:
            self.case["baseMVA"] = mvabase
        if version:
            self.case["version"] = version

        self.inputs : dict = {}
        """Time series mapped data inputs (see `pypower_sim.ppdata.PPData.set_input`)"""

        self.outputs : dict = {}
        """Time series mapped data outputs (see `pypower_sim.ppdata.PPData.set_output)"""

        self.recorders : dict = {}
        """Time series non-mapped data outputs (see `pypower_sim.ppdata.PPData.set_recorder)"""

        self.options : dict = dict(self.default_options)
        """Solver options"""

        self.errors : list = []
        """Solver errors detected"""

        self.profile : dict = None
        """Solver profile results"""

        self.cost : float = None
        """OPF cost result (if any)"""

    @staticmethod
    def get_header(name:str,*,ignore:list[str]=None) -> list[str]:
        """Convert idx data to a header list

        # Arguments

        - `idx`: module containing index values

        - `ignore`: list of index values to ignore

        # Returns

        - `list[str]`: ordered list of data array column header names
        """
        idx = globals()[f"idx_{name}"]
        if ignore is None:
            ignore = PPModel.standard_idx[name]
        mapping = {getattr(idx,x):x for x in dir(idx) if not x.startswith("_") and x not in ignore}
        indexes = sorted(mapping)
        assert max(indexes) - min(indexes) + 1 == len(indexes), \
            "indexes are not strictly sequential"
        return [mapping[n] for n in indexes]

    def to_dict(self) -> dict:
        """Convert model to a dict"""
        return {
            "application": "pypower_sim",
            "version": pkg_version("pypower_sim"),
            "name": self.name,
            "case": self.case,
            "inputs": {("|".join(x) if isinstance(x,(tuple,list)) else x):y
                for x,y in self.inputs.items()},
            "outputs": {("|".join(x) if isinstance(x,(tuple,list)) else x):y
                for x,y in self.outputs.items()},
            "recorders": {("|".join(x) if isinstance(x,(tuple,list)) else x):y
                for x,y in self.recorders.items()},
            "options": self.options,
            "errors": self.errors,
            "profile": self.profile,
        }

    def set_case(self,
        data:str|Callable|dict|None,
        ):
        """Set case data

        Arguments
        ---------

        - `data`: case data

        Description
        -----------

        Data can be a filename, a callable class or function, or a dictionary.
        If a filename is used, the module is loaded using the basename as the
        callable function. The callable function may be specified using the
        `call@file.py` syntax.
        """

        # default/convert data
        if data is None:
            data = copy(self.default_model)
        elif isinstance(data,str) and data.endswith(".py"):
            if self.name is None:
                self.name = os.path.splitext(os.path.basename(data))[0]
            callname,_ = data.split("@",1) if "@" in data else (self.name,data)
            modspec = spec_from_file_location(self.name,data)
            module = module_from_spec(modspec)
            sys.modules[self.name] = module
            modspec.loader.exec_module(module)
            data = getattr(module,callname)()
            # print(data)
        elif isinstance(data,str) and data.endswith(".json"):
            raise NotImplementedError("JSON not supported")
        elif callable(data):
            data = data()
        assert isinstance(data,dict), f"{data=} is not a dict"

        # read case data
        self.case = copy(self.default_model)
        self.case["version"] = data["version"]
        self.case["baseMVA"] = data["baseMVA"]
        for x in ["version","baseMVA","bus","branch","gen","gencost","dcline","dclinecost","gis"]:
            if x in data:
                self.case[x] = data[x]

        # check data
        assert "version" in self.case, "version missing in case"
        assert int(self.case["version"]) == int(self.default_model["version"]), \
        f"version={self.case['version']} is not valid"
        assert "baseMVA" in self.case, "baseMVA missing in case"
        assert self.case["baseMVA"] > 0.0, f"baseMVA={self.case['baseMVA']} is not valid"
        assert "bus" in self.case, "bus missing in case"
        assert "branch" in self.case, "branch missing in case"
        assert "gen" in self.case, "gen missing in case"
        if "gencost" in self.case:
            assert len(self.case["gencost"]) == len(self.case["gen"]), \
            f"len(gencost)={len(self.case['gencost'])} does not match" \
            f" len(gen)={len(self.case['gen'])}"

        # fix zero basekv values
        if "BASE_KV" in self.default_values and not self.default_values["BASE_KV"] is None:
            zerokv = self.case["bus"][:,idx_bus.BASE_KV] == 0
            if zerokv.any():
                self.case["bus"][zerokv,idx_bus.BASE_KV] = self.default_values["BASE_KV"]

        # gather zone basekv values
        if len(self.case["bus"]) > 0:
            self.kv_zones = sorted({float(x) for x in self.case["bus"][:,idx_bus.BASE_KV]})
            self.bus_kvzone = [self.kv_zones.index(x) for x in self.case["bus"][:,idx_bus.BASE_KV]]
        else:
            self.kv_zones = []
            self.bus_kvzone = []

    def _bus_i(self,x:int|list[int]) -> int|list[int]:
        """Get bus index from bus id

        Arguments
        ---------

        - `x`: bus id

        Returns
        -------

        - `int`: bus index

        - `list[int]`: list of bus indexes
        """
        if isinstance(x,int):
            result = self.case["bus"][:,idx_bus.BUS_I].astype(int).tolist().index(int(x))
        elif isinstance(x,list):
            result = [self._bus_i(y) for y in x]
        else:
            raise ValueError(f"{x=} is not an integer of list of integers")
        return result

    def get_refbus(self) -> list[int]:
        """Get list of reference busses

        Returns
        -------

        - `list[int]`: list of reference bus indexes
        """
        refs = [n for n,x in enumerate(self.case["bus"][:,idx_bus.BUS_TYPE]==3) if x]
        return self._bus_i(self.case["gen"][refs,idx_gen.GEN_BUS].astype(int).tolist())

    def from_dict(self,data:dict):
        """Convert dict to model

        # Arguments

        - `data`: source data
        """
        assert data["application"] == "pypower_sim", "JSON is not a pypower_sim model"
        assert data["version"] <= pkg_version("pypower_sim"), \
            "JSON model is from a newer version of pypower_sim, " \
            "you should update pypower_sim to at least that version"

        self.name = data["name"] if "name" in data else "unnamed"

        # extract case data and convert back to ndarray
        assert "case" in data, "JSON model does not contain a case"
        self.set_case(data["case"])

        self.inputs = {tuple(x.split("|")):PypowerModelDecoder(y)
            for x,y in data["inputs"].items()} if "inputs" in data else {}
        self.outputs = {x:PypowerModelDecoder(y)
            for x,y in data["outputs"].items()} if "outputs" in data else {}
        self.recorders = {x:PypowerModelDecoder(y)
            for x,y in data["recorders"].items()} if "recorders" in data else {}

        self.options = data["options"] if "options" in data else {}
        self.errors = data["errors"] if "errors" in data else []
        self.profile = data["profile"] if "profile" in data else None

    def to_json(self,*args,**kwargs) -> str|None:
        """Convert model to JSON

        # Arguments

        - `*args`: see json.dump()

        - `**kwargs`: see json.dump()

        # Returns

        - `str`: JSON data if no `fh` is specified

        - None: if `fh` is specified (JSON is write to file instead of returned)
        """
        return json.dump(self.to_dict(),*args,cls=PypowerModelEncoder,**kwargs)

    def from_json(self,*args,**kwargs):
        """Convert model from JSON"""
        self.from_dict(json.load(*args,**kwargs,object_hook=PypowerModelDecoder))

    def save(self,
        file:io.StringIO|str|None=None,
        **kwargs,
        ) -> str|None:
        """Save the model to a file

        # Arguments

        - `file`: file handle, name, or None to return data

        - `**kwargs`: JSON dump options

        # Returns

        - `str`: JSON data if `file` is `None`

        - None: if `file` is specified (JSON is write to file instead of returned)
        """

        if isinstance(file,str):
            with open(file,"w",encoding="utf-8") as fh:
                return self.save(fh,**kwargs)

        if file is None:
            return json.dumps(self.to_dict(),cls=PypowerModelEncoder,**kwargs)

        if hasattr(file,"writable") and file.writable():
            self.to_json(file,**kwargs)
            return None

        raise ValueError(f"{file=} is not writable")

    def load(self,
        file:io.StringIO|str|None=None,
        ):
        """Load the model from a file

        # Arguments

        - `file`: file handle, name, or None for stdin
        """
        if isinstance(file,str):
            with open(file,"r",encoding="utf-8") as fh:
                return self.load(fh)

        if file is None:
            return self.load(sys.stdin)

        if hasattr(file,"readable") and file.readable():
            self.from_json(file)
            return None

        raise ValueError(f"{file=} is not readable")

    def save_case(self,
        file:io.StringIO=sys.stdout,
        precision:int=9,
        name:str|None=None
        ):
        """Save the case data to a file

        # Arguments

        - `file`: file handle to which case data is saved

        - `precision`: float rounding precision
        """
        print(f"""# pypower case '{self.name}' saved on {dt.datetime.now()}
from numpy import array
def {self.name if not name else name}():
    return {{""",file=file)
        valid_keys = ["version","baseMVA","bus","branch","gen","gencost","dcline","dclinecost"]
        for key,value in [(x,y) for x,y in self.case.items() if x in valid_keys]:
            if isinstance(value,np.ndarray):
                print(f"""      '{key}': array([""",file=file)
                header = ",".join([f"{{0:>{precision+3}s}}".format(x)
                    for x in PPModel.get_header(key)])
                print(f"         #{header}",file=file)
                for row in value.tolist():
                    print(f"""         [{','.join([f'{{0:{precision+3}g}}'
                        .format(round(x,precision)) for x in row])}],""",file=file)
                print("        ]),",file=file)
            else:
                print(f"""      '{key}': {value},""",file=file)
        print("    }",file=file)

    def print(self,
        items:list[str]=None,
        file:[io.TextIOWrapper]=sys.stdout,
        flush:bool=True
        ):
        # pylint: disable=too-many-branches,too-many-locals
        """Print case data

        # Arguments

        - `items`: list of items to print

        - `file`: file to which items are printed
        """
        if items is None:
            items = self.standard_idx

        print(file=sys.stdout,flush=flush)
        print(file=sys.stderr,flush=flush)

        if "bus" in items:
            bus_cols = PPModel.get_header("bus")
            bus = pd.DataFrame(data=self.case["bus"],
                columns=bus_cols[:self.case["bus"].shape[1]],
                )
            for column,dtype in self.types_idx["bus"].items():
                bus[column] = bus[column].astype(dtype)
            bus.index.name="BUS"
            print(bus,file=file)

        if "branch" in items:
            branch_cols = PPModel.get_header("branch")
            branch = pd.DataFrame(data=self.case["branch"],
                columns=branch_cols[:self.case["branch"].shape[1]])
            for column,dtype in self.types_idx["branch"].items():
                branch[column] = branch[column].astype(dtype)
            branch.index.name="BRANCH"
            print(branch,file=file)

        if "gen" in items:
            gen_cols = PPModel.get_header("gen")
            gen = pd.DataFrame(data=self.case["gen"],
                columns=gen_cols[:self.case["gen"].shape[1]])
            for column,dtype in self.types_idx["gen"].items():
                gen[column] = gen[column].astype(dtype)
            gen.index.name="GEN"
            print(gen,file=file)

        if "gencost" in items and "gencost" in self.case:
            cost_cols = PPModel.get_header("gencost")
            ncost = self.case["gencost"].shape[1] - len(cost_cols)
            cost_cols.extend([f"COST{n+1}" for n in range(int(ncost))])
            gencost = pd.DataFrame(data=self.case["gencost"],columns=cost_cols)
            gencost.rename({"COST":"COST0"},inplace=True,axis=1)
            for column,dtype in self.types_idx["gencost"].items():
                gencost[column] = gencost[column].astype(dtype)
            gencost.index.name="GENCOST"
            print(gencost,file=file)

        if "dcline" in items and "dcline" in self.case and len(self.case["dcline"]) > 0:
            dcline_cols = PPModel.get_header("dcline")
            dcline = pd.DataFrame(data=self.case["dcline"],
                columns=dcline_cols[:self.case["dcline"].shape[1]])
            for column,dtype in self.types_idx["dcline"].items():
                dcline[column] = dcline[column].astype(dtype)
            dcline.index.name="DCLINE"
            print(dcline,file=file)

        if "dclinecost" in items and "dclinecost" in self.case and len(self.case["dclinecost"]) > 0:
            cost_cols = PPModel.get_header("dclinecost")
            ncost = self.case["dclinecost"].shape[1] - len(cost_cols)
            cost_cols.extend([f"COST{n+1}" for n in range(int(ncost))])
            dclinecost = pd.DataFrame(data=self.case["dclinecost"],columns=cost_cols)
            dclinecost.rename({"COST":"COST0"},inplace=True,axis=1)
            for column,dtype in self.types_idx["dclinecost"].items():
                dclinecost[column] = dclinecost[column].astype(dtype)
            dclinecost.index.name="DCLINECOST"
            print(dclinecost,file=file)

        print(file=sys.stdout,flush=flush)
        print(file=sys.stderr,flush=flush)

    def save_kml(self,
        filename:str,
        use_geocode:bool=False,
        ):
        """Generate KML output

        # Arguments

        - `filename`: KML filename of output (see `pypower_sim.kml.KML`)

        - `use_geocode`: marker names are geocode instead of bus id
        """
        kml = KML(filename)

        # bus style
        kml.add_markerstyle(
            name="node",
            url="https://maps.google.com/mapfiles/kml/pal3/icon49.png",
            )

        # bus markers
        for bus_i,latitude,longitude,geocode in self.case["gis"][:,0:4]:
            kml.add_marker(
                name=geocode if use_geocode else f"{bus_i}",
                style="node",
                position=[longitude,latitude,0.0],
                )

        # line style
        kml.add_linestyle(
            name="line-in",
            color="7f00ffff",
            width=4,
            )
        kml.add_linestyle(
            name="line-out",
            color="7f000000",
            width=4,
            )

        # line paths
        gis = {n:(y,x,0,c) for n,x,y,c in self.case["gis"][:,:4]}
        for data in self.case["branch"]:
            fbus = int(data[idx_branch.F_BUS])
            tbus = int(data[idx_branch.T_BUS])
            status = int(data[idx_branch.BR_STATUS])
            kml.add_line(
                name=f"{fbus}-{tbus}",
                style="line-in" if status else "line-out",
                from_position=gis[fbus][0:3],
                to_position=gis[tbus][0:3],
                )
        for data in self.case["dcline"]:
            fbus = int(data[idx_branch.F_BUS])
            tbus = int(data[idx_branch.T_BUS])
            status = int(data[idx_branch.BR_STATUS])
            kml.add_line(
                name=f"{fbus}-{tbus}",
                style="line-in" if status else "line-out",
                from_position=gis[fbus][0:3],
                to_position=gis[tbus][0:3],
                )

        kml.close()

    def perunit(self,
        unit:str,
        kind:str="bus",
        ref:int=None,
        ):
        """Per-unit base values

        Arguments
        ---------

        - `unit`: unit to scale (`MW`, `V`, `Ohm`, `S`, or `A)

        - `kind`: per-unit vector kind (`bus` or `branch`)

        - `ref`: reference bus to use (default to swing bus)

        Returns
        -------
        """
        if ref is None:
            ref = self.get_refbus()
        if isinstance(ref,list):
            assert len(ref) == 1, f"only one reference bus can be specified ({ref=})"
            ref = ref[0]
        assert isinstance(ref,int), f"reference bus must be an integer ({ref=})"
        if kind == "bus":
            # pylint: disable=invalid-name
            Sbase = self.case["baseMVA"]
            Vbase = self.case["bus"][ref,idx_bus.BASE_KV]
            match unit:
                case 'MW':
                    return Sbase
                case 'V':
                    return Vbase
                case 'Ohm':
                    return Vbase**2 / Sbase
                case 'S':
                    return Sbase / Vbase**2
                case 'A':
                    return Sbase / Vbase / np.sqrt(3)
            raise ValueError(f"{unit=} is invalid")

        if kind == "branch":

            raise NotImplementedError("{kind=} is not implemented yet")

        return None

    bus_optional = ["LAM_P","LAM_Q","MU_VMIN","MU_VMAX"]
    @classmethod
    def bus(cls,**kwargs) -> np.array:
        """Create bus data

        # Arguments

        - `kwargs`: merged bus, load, and shunt data (see `pypower.idx_bus` for
          details)

        # Returns

        - `numpy.array`: bus data
        """

        header = PPModel.get_header("bus")
        for key,value in kwargs.items():
            if key not in header:
                raise KeyError(f"{key}={value} is not a valid bus item")

        result = []
        for item in header:
            if item in kwargs:
                result.append(kwargs[item])
            elif item not in cls.bus_optional:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    branch_optional = ["PF","PT","QF","QT","MU_SF","MU_ST","MU_ANGMIN","MU_ANGMAX"]
    @classmethod
    def branch(cls,**kwargs) -> np.array:
        """Create branch data

        # Arguments

        - `kwargs`: merged branch and transformer data (see `pypower.idx_brch` for
          details)

        # Returns

        - `numpy.array`: bus data
        """
        header = PPModel.get_header("branch")
        for key,value in kwargs.items():
            if key not in header:
                raise KeyError(f"{key}={value} is not a valid branch item")

        result = []
        for item in header:
            if item in kwargs:
                result.append(kwargs[item])
            elif item not in cls.branch_optional:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    gen_optional = ["MU_PMAX","MU_PMIN","MU_QMAX","MU_QMIN"]
    @classmethod
    def gen(cls,**kwargs) -> np.array:
        """Create gen data

        # Arguments

        - `kwargs`: generation data (see `pypower.idx_gen` for details)

        # Returns

        - `numpy.array`: gen data
        """

        result = []
        for item in PPModel.get_header("gen"):
            if item in kwargs:
                result.append(kwargs[item])
            elif item not in cls.gen_optional:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    @classmethod
    def gencost(cls,**kwargs) -> np.array:
        """Create gencost data

        # Arguments

        - `kwargs`: generation data (see `pypower.idx_gen` for details)

        # Returns

        - `numpy.array`: cost data
        """

        result = []
        for item in PPModel.get_header("gencost"):
            if item in kwargs:
                if kwargs[item].ndim == 1:
                    result.append(kwargs[item])
                else:
                    for col in range(kwargs[item].shape[1]):
                        result.append(kwargs[item][:,col])
            else:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    dcline_optional = ["MU_PMIN","MU_PMAX","MU_QMINF","MU_QMAXF","MU_QMINT","MU_QMAXT"]
    @classmethod
    def dcline(cls,**kwargs) -> np.array:
        """Create dcline data

        # Arguments

        - `kwargs`: dcline data (see `pypower.idx_dcline` for details)

        # Returns

        - `numpy.array`: dcline data
        """
        result = []
        for item in PPModel.get_header("dcline"):
            if item in kwargs:
                result.append(kwargs[item])
            elif item not in cls.dcline_optional:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    @classmethod
    def dclinecost(cls,**kwargs) -> np.array:
        """Create dclinecost data

        # Arguments

        - `kwargs`: dclinecost data (see `pypower.idx_cost` for details)

        # Returns

        - `numpy.array`: cost data
        """

        result = []
        for item in PPModel.get_header("dclinecost"):
            if item in kwargs:
                if kwargs[item].ndim == 1:
                    result.append(kwargs[item])
                else:
                    for col in range(kwargs[item].shape[1]):
                        result.append(kwargs[item][:,col])
            else:
                raise ValueError(f"missing {item} data")

        return np.array(result)

    def get_info(self) -> dict:
        """Get model information

        # Returns

        - `dict`: table of model information
        """
        bus = self.get_data("bus")
        gengis = pd.merge(
                self.get_gis(),
                self.get_data("gen"),
                left_on="BUS_I",
                right_on="GEN_BUS")
        loadgis = pd.merge(
                self.get_gis(),
                self.get_data("bus"),
                left_on="BUS_I",
                right_on="BUS_I")
        return {
            "Model name": self.name,
            "Bus count": len(bus),
            "Branch count": len(self.get_data("branch")),
            "Generator count": len(self.get_data("gen")),
            "DC line count": len(self.get_data("dcline")),
            "Node count": len(self.get_nodes()),
            "LV busses": len(bus[bus.BASE_KV==20]),
            "MV busses": len(bus[(bus.BASE_KV>20)&(bus.BASE_KV<250)]),
            "HV busses": len(bus[bus.BASE_KV>250]),
            "Generation substations": len(gengis.GEOHASH.unique()),
            "Load substations": len(loadgis[loadgis.PD>0].GEOHASH.unique()),
            }

    def get_data(self,name) -> pd.DataFrame:
        """Get data table with data types

        # Arguments

        - `name`: name of case data to return (e.g., `"bus"`,`"branch"`,`"gis"`, etc.)

        # Returns

        - `pandas.DataFrame`: case data requested with data types (see `types_idx`)
        """
        assert name in self.standard_idx, f"'{name}' is not a valid data item name"
        width = self.case[name].shape[1]
        header = PPModel.get_header(name)
        n = 1
        last = header[-1]
        if len(header) < width:
            header[-1] = f"{last}0"
        while len(header) < width:
            header.append(f"{last}{n}")
            n += 1
        result = pd.DataFrame(self.case[name].T,header[:width]).T
        for column,dtype in self.types_idx[name].items():
            result[column] = result[column].astype(dtype)
        return result

    def get_gis(self) -> pd.DataFrame:
        """Get indexed GIS data

        # Returns

        - `pandas.DataFrame`: case GIS data (no index, sorted by row number)
        """
        return self.get_data("gis").reset_index().sort_index()

    def get_bus(self,
        bustype:int|None=None,
        index:str|None=None,
        ) -> pd.DataFrame:
        """Get data for all load busses

        # Arguments

        - `bustype`: bus type of get (i.e., `idx_bus.PQ`, `idx_bus.PV`, `idx_bus.REF`)

        - `index`: index to use (merge with GIS data if index in GIS columns)

        # Returns

        - `pandas.DataFrame`: bus data
        """
        bus = self.get_data("bus")
        if index in self.get_header("bus"):
            bus.set_index(index,inplace=True)
        elif index in self.get_header("gis"):
            gis = self.get_data("gis")
            bus = pd.merge(bus,gis,left_on="BUS_I",right_on="BUS_I").set_index(index)
        elif not index is None:
            raise ValueError(f"{index=} is invalid")
        return bus if bustype is None else bus[bus.BUS_TYPE==bustype]

    def get_nodes(self,data:pd.DataFrame|None=None) -> dict:
        """Get a dictionary of nodes and their busses

        # Arguments

        - `data`: dataframe from use (`"gis"` if None)

        # Returns

        - `dict`: table of nodes and bus ids
        """
        nodes = {}
        for _,row in (self.get_gis() if data is None else data)[
                ["BUS_I","GEOHASH"]].iterrows():
            bus_i,geohash = row.values
            if geohash in nodes:
                nodes[geohash].append(bus_i)
            else:
                nodes[geohash] = [bus_i]
        return nodes

    def get_graph(self,
        level:str="BUS",
        nodes:str=None
        ) -> tuple[pd.DataFrame,pd.DataFrame]:
        """Get network graphs

        # Arguments

        - `level`: `"BUS"`, `"NODE"`, `"ZONE"`, or `"AREA"`

        - `nodes`: return node type (`None`, `"nearest"`, `"centroid"`)

        # Returns

        - `links`: list of link tuples indexes into bus data in order of
          branch data
        """
        nodes = pd.merge(self.get_data("bus"),
            self.get_data("gis"),
            left_on="BUS_I",
            right_on="BUS_I",
            )

        match level:

            case "BUS":

                # no node aggregation
                nodes["BUS"] = nodes.index # save row indexing for linklist
                nodes.set_index("BUS_I",inplace=True) # index on bus id

                # process branches
                branch = self.get_data("branch").copy()
                branch["FROM"] = nodes.loc[branch.F_BUS.astype(int)].BUS.values
                branch["TO"] = nodes.loc[branch.T_BUS.astype(int)].BUS.values
                branch.set_index(["FROM","TO"],inplace=True)

                # process dclines
                dcline = self.get_data("dcline").copy()
                dcline["FROM"] = nodes.loc[dcline.F_BUS.astype(int)].BUS.values
                dcline["TO"] = nodes.loc[dcline.T_BUS.astype(int)].BUS.values
                dcline.set_index(["FROM","TO"],inplace=True)

                links = branch.index.tolist() + dcline.index.tolist()

            case "GEOHASH":

                warnings.warn(f"{level=} not implemented yet")
                links = pd.DataFrame({"FROM":["-1"],"TO":["-1"]})\
                    .set_index(["FROM","TO"]).index # TODO: write graph at geohash level

            case "ZONE":

                warnings.warn(f"{level=} not implemented yet")
                links = pd.DataFrame({"FROM":["-1"],"TO":["-1"]})\
                    .set_index(["FROM","TO"]).index # TODO: write graph at zone level

            case "AREA":

                warnings.warn(f"{level=} not implemented yet")
                links = pd.DataFrame({"FROM":["-1"],"TO":["-1"]})\
                    .set_index(["FROM","TO"]).index # TODO: write graph at area level

            case "_":

                raise ValueError(f"{level=} is invalid")

        linklist = [[int(y) for y in x] for x in links]
        return linklist

    def get_violations(self,
        abs_err : float = None, # absolute error
        rel_err : float = None # relative error
        ) -> dict[str:list[str]]:
        # pylint: disable=too-many-branches
        """Get list of powerflow case violations

        Arguments
        ---------

        - `precision`: rounding precision to use when comparion actual to limits and ratings

        Returns
        -------

        dict[]
        """
        Fuzzy.abs_err = abs_err
        Fuzzy.rel_err = rel_err

        result = []

        # check busses
        for n,bus in self.get_data("bus").iterrows():

            # check bus voltage magnitudes
            if not Fuzzy(bus.VMIN) <= bus.VM and Fuzzy(bus.VM) <= bus.VMAX:
                result.append(f"bus#{n}: bus '{bus.BUS_I:.0f}' VM={bus.VM:.3f}"\
                    f" puV outside limits ({bus.VMIN:.3f},{bus.VMAX:.3f}) puV")

        # check branches
        for n,branch in self.get_data("branch").iterrows():

            if branch.BR_STATUS == 0:
                continue

            # check branch flow limits
            if not "PF" in branch or not "QF" in branch:
                continue
            flow = max(abs(complex(branch.PF,branch.QF)),abs(complex(branch.PT,branch.QT)))
            rated = 0.0 if branch.BR_STATUS == 0 else max(branch.RATE_A,branch.RATE_B,branch.RATE_C)
            if rated > 0 and Fuzzy(flow) > rated:
                result.append(f"branch#{n}: branch from bus '{branch.F_BUS:.0f}' "
                    f"to '{branch.T_BUS:.0f}' apparent power flow {flow:.3f} "
                    f"MVA exceeds maximum rating {rated:.3f} MW")

        # check generators
        for n,gen in self.get_data("gen").iterrows():

            # ignore generation that are not running or have invalid/no limits
            if gen.GEN_STATUS == 0:
                continue

            # check real power
            if gen.PG > 0 and gen.PMAX > gen.PMIN \
                    and not Fuzzy(gen.PMIN) <= gen.PG \
                    and not Fuzzy(gen.PG) <= gen.PMAX:
                result.append(f"gen#{n}: generator at bus '{gen.GEN_BUS:.0f}' "
                    f"real power PG={gen.PG:.3f} MW outside power "
                    f"dispatch limits ({gen.PMIN:.3f},{gen.PMAX:.3f}) MW")

            # check reactive power
            if gen.QG != 0 and gen.QMAX < gen.QMIN \
                    and not Fuzzy(gen.QMIN) <= gen.QG \
                    and Fuzzy(gen.QG) <= gen.QMAX:
                result.append(f"gen#{n}: generator at bus '{gen.GEN_BUS:.0f}' "
                    f"reactive power QG={gen.QG:.3f} MVAr outside power "
                    f"dispatch limits ({gen.QMIN:.1f},{gen.QMAX:.3f}) MVAr")

        # check DC lines
        if "dcline" in self.case:
            for n,dcline in self.get_data("dcline").iterrows():

                if dcline.BR_STATUS == 0:
                    continue

                # check real power limits
                if Fuzzy(dcline.PF) > dcline.PMAX:
                    result.append(f"dcline#{n}: DC line real power "
                        f"{dcline.PF:.3f} MW exceeds maximum {dcline.PMAX:.3f} MW")
                if Fuzzy(dcline.PF) < dcline.PMIN:
                    result.append(f"dcline#{n}: DC line real power {dcline.PF:.3f} "
                        f"MW below minimum {dcline.PMIN:.3f} MW")

                # check reactive power limits
                if not Fuzzy(dcline.QMINF) <= dcline.QF and Fuzzy(dcline.QF) <= dcline.QMAXF:
                    result.append(f"dcline#{n}: DC line reactive power "
                        f"{dcline.QF:.3f} MVAr at 'from' end outside limits "
                        f"({dcline.QMINF:.3f},{dcline.QMAXF:.3f}) MVAr")
                if not Fuzzy(dcline.QMINT) <= dcline.QT and Fuzzy(dcline.QT) <= dcline.QMAXT:
                    result.append(f"dcline#{n}: DC line reactive power "
                        f"{dcline.QT:.3f} MVAr at 'to' end outside limits "
                        f"({dcline.QMINT:.3f},{dcline.QMAXT:.3f}) MVAr")

        return result

    # package-only utility methods
    def _data(self):
        return PPData(self)

    def _plots(self):
        return PPPlots(self)

    def _solver(self):
        return PPSolver(self)

    def _graph(self):
        return PPGraph(self)

if __name__ == '__main__':

    from ppsolver import PPSolver

    pd.options.display.max_columns = None
    pd.options.display.width = None
    pd.options.display.max_rows = None

    tests = sorted([x[4:-3] for x in os.listdir("../test") \
        if x.startswith("case") and x.endswith(".py")])
    errors = 0
    for caseid in tests:

        print(f"Running case{caseid}.py",end="... ",flush=True)
        try:

            test_case = f"../test/case{caseid}.py"
            test = PPModel(os.path.splitext(os.path.basename(test_case))[0],case=test_case)
            solver = PPSolver(test)

        # pylint: disable=broad-exception-caught
        except Exception as err:

            print(f"EXCEPTION [test/case{caseid}.py]: {err}, skipping it")
            errors += 1
            continue

        if "gencost" in test.case:
            print("OPF",flush=True,end=" ")
            solver.solve_opf(use_acopf=True)
            violations = test.get_violations()
            if violations:
                print("ERROR","","AC OPF SOLUTION","---------------",sep="\n")
                test.print()
                print("","OPF VIOLATIONS","--------------",*violations,sep="\n")
                print(f"...continuing case{caseid}.py",end="... ")
                errors += 1
            else:
                print("ok",end="... ",flush=True)

            print("PF",flush=True,end=" ")
            solver.solve_pf()
            violations = test.get_violations()
            if violations:
                print("ERROR","","POWERFLOW SOLUTION","------------------",sep="\n")
                test.print()
                print("","PF VIOLATIONS","-------------",*violations,sep="\n")
                print(f"...continuing case{caseid}.py",end="... ")
                errors += 1
            else:
                print("ok",end="... ",flush=True)

            print("done")
    if errors:
        print(f"{errors} errors found!")
    else:
        print("No errors")
