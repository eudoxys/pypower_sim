"""PyPOWER Data I/O

The PyPOWER model data manager controls the flow of data in and out of a
pypower model.

Example:

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

        Arguments:

        model: """
        self.model = model

    def set_input(self,
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        name:str,
        column:str,
        file:str,
        scale:float=1.0,
        offset:float=0.0,
        mapping:dict=None,
        ):
        """Set a timeseries input data feed

        Arguments:

        name: data set name (e.g., bus, branch)

        column: data column name (e.g., "PD")

        file: file name from which data is input

        scale: scaling factor to apply to input data

        offset: offset to apply to the scaled data

        mapping: maps column names to data rows with weights
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

        Arguments:

        name: data set name (e.g., bus, branch)

        column: data column name (e.g., "PD)

        file: file name to which data is output

        scale: scaling factor to apply to output data

        offset: offset to apply to scaled data

        mapping: maps column names to data rows with weights

        formatting: formatting of output
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

        file: file name to which data is output

        name: output column name

        target: case keys to value to record (e.g., ["cost"])

        scale: scaling factor to apply to output data

        offset: offset to apply to scaled data

        format: formatting of output
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
