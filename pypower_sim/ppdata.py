"""PyPOWER Data I/O

The PyPOWER model data manager controls the flow of data in and out of a
pypower model.

# Description

There are four types of data I/O available:

1. Inputs

    Inputs are the preferred method of reading large amounts of structured data
    into a single case structure column. Inputs read CSV files that are
    arranged with columns that map to multiple rows in a single target case
    data structure. For example, a CSV file that contains time-series values
    for the load values of of busses would be handled by the
    `pypower_sim.ppdata.PPPlot.set_input` method to read column values to the
    `PD` and `QD` properties of busses.  

2. Players

    Players are the preferred method of reading unstructured data into
    individual properties of cases. Players read CSV files that are arranged
    with columns that map to properties of target case data structures. For
    example, CSV file that contains time-series values for the constraints on
    line flows would be handled by the `pypower_sim.ppdate.PPPlot.set_player`
    method read column values to the `RATE_A`, `RATE_B`, and `RATE_C`
    properties of specified branches.

3. Outputs

    Outputs write CSV files in a manner similar to #Inputs, i.e., sampling a
    property from rows of a case object to columns of a CSV file. This is the
    preferred method to generate large amounts of structure data from cases.

4. Recorders

    Recorder write CSV files in a manner similar to #Players, i.e., sampling
    individual case data properties and writing them to a CSV file. This is the
    preferred method to generate unstructured data from individual properties
    of cases.

# Example

    from wecc240 import wecc240
    model = PPModel(case=wecc240)

    datamgr = PPData(model)
    datamgr.set_input("bus","PD","bus_PD.csv",scale=10)
    datamgr.set_output("bus","VA","bus_VA.csv")
    datamgr.set_recorder("cost.csv","cost",["cost"])
    
    solver = PPSolver(model)
    solver.run_timeseries(
        start="2020-08-01 00:00:00+07:00",
        end="2020-09-01 00:00:00+07:00",
        freq="1h",
        )
"""

import os
from typing import TypeVar
import pandas as pd
import numpy as np

class PPData:
    """PyPOWER model data I/O manager"""

    def __init__(self,model=TypeVar('PPModel')):
        """Setup data I/O manager

        # Arguments

        - `model`: `pypower_sim.ppmodel.PPModel` object

        # See also

        - `pypower_sim.ppmodel.PPModel.inputs`
        - `pypower_sim.ppmodel.PPModel.outputs`
        - `pypower_sim.ppmodel.PPModel.recorders`
        """
        
        self.model = model
        """`pypower_sim.ppmodel.PPModel` object"""

    def set_input(self,
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        name:str,
        column:str,
        file:str,
        scale:float=1.0,
        offset:float=0.0,
        mapping:dict=None,
        ):
        """Set a time-series input data feed

        # Arguments

        - `name`: data set name (e.g., bus, branch)

        - `column`: data column name (e.g., "PD")

        - `file`: file name from which data is input

        - `scale`: scaling factor to apply to input data

        - `offset`: offset to apply to the scaled data

        - `mapping`: maps column names to data rows with weights

        # See also

        - `pypower_sim.ppmodel.PPModel.inputs`
        """
        assert name in self.model.standard_idx,f"{name=} is not valid"
        assert column in self.model.get_header(name), f"{column=} is not found in {name} data"
        assert (name,column) not in self.model.inputs, f"input({name=},{column=}) already defined"
        if file is None:
            del self.model.inputs[name]
        else:
            assert os.path.exists(file), f"{file=} not found"
            data = pd.read_csv(file,index_col=[0],parse_dates=[0]) * scale + offset
            data.index.name = "datetime"

            # default to direct mapping of column names to row numbers
            if mapping is None:
                mapping = {
                    "index": data.columns.astype(int),
                    "scale": np.ones(len(data.columns)),
                    }

            # set up input
            self.model.inputs[(name,column)] = {
                "file": file,
                "data": data,
                "mapping": mapping,
            }

    def set_player(self,
        # pylint: disable=too-many-arguments,too-many-position-arguments
        name:str,
        columns:dict[str:str],
        file:str,
        scale:float=1.0,
        offset:float=0.0,
        ):
        """Set a time-series player input

        # Arguments

        - `name`: data set name (e.g., bus, branch)

        - `columns`: data columns name mapping (e.g., "PD")

        - `file`: file name from which data is input

        - `scale`: scaling factor to apply to input data

        - `offset`: offset to apply to the scaled data
        """
        assert name in self.model.standard_idx,f"{name=} is not valid"
        notfound = set(columns.values) - set(self.model.get_header(name))
        assert notfound == set() , f"columns {notfound} is not found in {name} data"
        assert (name,list(columns)) not in self.model.inputs, f"input({name=},{list(columns)=}) already defined"
        if file is None:
            del self.model.inputs[name]
        else:
            assert os.path.exists(file), f"{file=} not found"
            data = pd.read_csv(file,index_col=[0],parse_dates=[0]) * scale + offset
            data.index.name = "datetime"

            # default to direct mapping of column names to row numbers
            if mapping is None:
                mapping = {
                    "index": data.columns.astype(int),
                    "scale": np.ones(len(data.columns)),
                    }

            # set up input
            self.model.inputs[(name,column)] = {
                "file": file,
                "data": data,
                "mapping": mapping,
            }

    def set_output(self,
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        name:str,
        column:str,
        file:str,
        scale:float=1.0,
        offset:float=0.0,
        mapping:dict=None,
        formatting:str="g"):
        """Set a timeseries output data feed

        # Arguments

        - `name`: data set name (e.g., bus, branch)

        - `column`: data column name (e.g., "PD)

        - `file`: file name to which data is output

        - `scale`: scaling factor to apply to output data

        - `offset`: offset to apply to scaled data

        - `mapping`: maps column names to data rows with weights

        - `formatting`: formatting of output

        # See also

        -  `pypower_sim.ppmodel.PPModel.outputs`
        """
        assert name in self.model.standard_idx, f"{name=} is not valid"
        assert column in self.model.get_header(name), f"{column=} is not found in {name} data"
        assert file not in self.model.outputs, f"{file=} already exists in the outputs"
        if mapping is None:
            nrows = len(self.model.case[name])
            mapping = {
                "rows": np.arange(nrows,dtype=int),
                "columns": [f"{x}" for x in range(nrows)],
                "scale": np.full(nrows,scale),
                "offset": np.full(nrows,offset),
            }
        # pylint: disable=consider-using-with
        self.model.outputs[file] = {
            "name": name,
            "column": column,
            "fh":None,
            "mapping": mapping,
            "format": formatting,
            }
        # print("timestamp",*mapping["columns"],sep=",",file=fh)

    def set_recorder(self,
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        file:str,
        name:str,
        target:list[str],
        scale:float=1.0,
        offset:float=0.0,
        formatting="g"):
        """Set a recorder

        # Arguments
        
        - `file`: file name to which data is output

        - `name`: output column name

        - `target`: case keys to value to record (e.g., ["cost"])

        - `scale`: scaling factor to apply to output data

        - `offset`: offset to apply to scaled data

        - `format`: formatting of output

        # See also

        -  `pypower_sim.ppmodel.PPModel.recorders`
        """
        assert isinstance(target,list), "target must be a list"
        assert all(isinstance(x,str) for x in target), "target must be a list of strings"
        if not file in self.model.recorders:
            # pylint: disable=consider-using-with
            self.model.recorders[file] = {
                "fh": None,
                "targets": {}
                }

        recorder = self.model.recorders[file]
        assert name not in recorder["targets"], f"target {name=} is already specified in {file=}"
        recorder["targets"][name] = {
            "source":target,
            "format":formatting,
            "transform": f"lambda x: x*{scale}+{offset}",
            }
