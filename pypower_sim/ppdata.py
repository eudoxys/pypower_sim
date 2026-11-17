"""PyPOWER Data I/O

The PyPOWER model data manager controls the flow of data in and out of a
pypower model.

# Description

There are four types of data I/O available:

1. Input panels

    A panel is a data frame whose rows are timestamps and whose columns are
    objects, i.e., rows of a single case data table (e.g., `bus`). Each panel
    feeds one property (column) of that table, e.g., `PD`. Column names are
    case row numbers unless a `mapping` is given to map them to rows. Panels
    are the preferred method of reading large amounts of structured data
    into a single case data column. For example, a CSV file that contains
    time-series values for the loads of busses is read with
    `pypower_sim.ppdata.PPData.set_input` to update the `PD` property of
    those busses. See `pypower_sim.ppdata.PPData.set_input`.

2. Players

    A player is a data frame whose rows are timestamps and whose columns are
    properties of the player's parent object, i.e., one row of a case data
    table. Players are the preferred method of reading unstructured data
    into individual properties of case objects. For example, a CSV file that
    contains time-series values for the ratings of a line is read with
    `pypower_sim.ppdata.PPData.set_player` to update the `RATE_A`, `RATE_B`,
    and `RATE_C` properties of that branch. A column can also name another
    object of the same table using `OBJECTNAME.PROPERTYNAME`, or an object
    of another table using `TABLE:OBJECTNAME.PROPERTYNAME`. See
    `pypower_sim.ppdata.PPData.set_player` for the column name syntax.

3. Outputs

    Outputs write CSV files in a manner similar to input panels, i.e.,
    sampling a property from rows of a case object to columns of a CSV file.
    This is the preferred method to generate large amounts of structure data
    from cases.

4. Recorders

    Recorder write CSV files in a manner similar to players, i.e., sampling
    individual case data properties and writing them to a CSV file. This is
    the preferred method to generate unstructured data from individual
    properties of cases.

# Example

    from wecc240 import wecc240
    model = PPModel(case=wecc240)

    datamgr = PPData(model)
    datamgr.set_input("bus","PD","bus_PD.csv",scale=10)
    datamgr.set_player("branch","branch_ratings.csv",parent=3)
    datamgr.set_output("bus","VA","bus_VA.csv")
    datamgr.set_recorder("cost.csv","cost",["cost"])

    solver = PPSolver(model)
    solver.run_timeseries(
        start="2020-08-01 00:00:00+07:00",
        end="2020-09-01 00:00:00+07:00",
        freq="1h",
        )

where `branch_ratings.csv` contains the ratings of branch row 3, the
`RATE_A` rating of branch row 4, and the `PMAX` of generator row 2:

    datetime,RATE_A,RATE_B,RATE_C,4.RATE_A,gen:2.PMAX
    2020-08-01 07:00:00+00:00,250,250,250,300,180
    2020-08-01 08:00:00+00:00,240,240,240,290,170
"""

import os
from typing import TypeVar
import pandas as pd
import numpy as np

PLAYER_PARTS = ("real","imag")
"""Parts of a property that a player column can refer to"""

class PPData:
    """PyPOWER model data I/O manager"""

    def __init__(self,model=TypeVar('PPModel')):
        """Setup data I/O manager

        # Arguments

        - `model`: `pypower_sim.ppmodel.PPModel` object

        # See also

        - `pypower_sim.ppmodel.PPModel.inputs`
        - `pypower_sim.ppmodel.PPModel.players`
        - `pypower_sim.ppmodel.PPModel.outputs`
        - `pypower_sim.ppmodel.PPModel.recorders`
        """

        self.model = model
        """`pypower_sim.ppmodel.PPModel` object"""

    def set_input(self,
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        name:str,
        column:str|None,
        file:str|None,
        scale:float=1.0,
        offset:float=0.0,
        mapping:dict|str|tuple[str,type]=None,
        ):
        """Set or remove a time-series input panel

        A panel is a CSV file whose first column holds the timestamps (UTC)
        and whose other columns are objects, i.e., rows of the `name` case
        data table. Every value in a row is written to the `column` property
        of the object named by its column, as `value * scale + offset`.

        Column names are case row numbers (not bus ids) unless `mapping` is
        given, e.g., this panel sets `PD` of bus rows 1, 2, 3, and 6:

            datetime,1,2,3,6
            2020-07-31 17:00:00+0000,10,20,30,40

        # Arguments

        - `name`: data set name (e.g., bus, branch)

        - `column`: data column name (e.g., "PD"); may be `None` only when
          `file` is `None`, to remove all the inputs of `name`

        - `file`: file name from which data is input, or `None` to remove the
          input of `(name,column)` (or every input of `name` when `column` is
          `None`)

        - `scale`: scaling factor to apply to input data

        - `offset`: offset to apply to the scaled data

        - `mapping`: maps column names to data rows with weights (a dict with
          `index` and `scale` lists), the name of a column of the `name` data
          whose values are the column names (e.g., `"BUS_I"`), or
          `(column,dtype)` to convert those values before matching (default
          dtype is `str`)

        # Exceptions

        - `AssertionError`: invalid table or column, missing file, input
          already defined, or removal of an input that is not defined

        # See also

        - `pypower_sim.ppmodel.PPModel.inputs`

        - `pypower_sim.ppdata.PPData.set_player`
        """
        assert name in self.model.standard_idx,f"{name=} is not valid"
        if file is None:
            if column is None:
                for key in [x for x in self.model.inputs if x[0] == name]:
                    del self.model.inputs[key]
            else:
                assert (name,column) in self.model.inputs, \
                    f"input({name=},{column=}) is not defined"
                del self.model.inputs[(name,column)]
        else:
            assert column in self.model.get_header(name), \
                f"{column=} is not found in {name} data"
            assert (name,column) not in self.model.inputs, \
                f"input({name=},{column=}) already defined"
            assert os.path.exists(file), f"{file=} not found"
            data = pd.read_csv(file,index_col=[0],parse_dates=[0]) * scale + offset
            data.index.name = "datetime"

            # default to direct mapping of column names to row numbers
            if mapping is None:
                mapping = {
                    "index": data.columns.astype(int),
                    "scale": np.ones(len(data.columns)),
                    }
            elif isinstance(mapping,(str,tuple)): # map using column data
                if isinstance(mapping,tuple):
                    mapping,dtype = mapping
                else:
                    dtype = str
                target = self.model.get_data(name)
                assert mapping in target.columns, f"{mapping=} is not a valid columns name in {name} data"
                maplist = data.columns.astype(dtype).values
                mapdata = list(target[mapping].astype(dtype).values)
                mapping = {
                    "index": [mapdata.index(x) for x in maplist],
                    "scale": np.ones(len(data.columns))
                }

            # set up input
            self.model.inputs[(name,column)] = {
                "file": file,
                "data": data,
                "mapping": mapping,
            }

    def set_player(self, # pylint: disable=too-many-locals
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        name:str,
        file:str,
        parent:int|str=None,
        scale:float=1.0,
        offset:float=0.0,
        mapping:dict|str|tuple[str,type]=None,
        mappings:dict[str,dict|str|tuple[str,type]]=None,
        ):
        """Set a time-series player

        A player is a CSV file whose first column holds the timestamps (UTC)
        and whose other columns are properties of the player's parent object,
        i.e., one row of the `name` case data table. Every value in a row is
        written to its property as `value * scale + offset`.

        Column names use the following syntax:

        - `PROPERTYNAME`: property of the parent object (e.g., `RATE_A`)

        - `OBJECTNAME.PROPERTYNAME`: property of the object `OBJECTNAME`
          instead of the parent (e.g., `4.RATE_A`)

        - `OBJECTNAME.PROPERTYNAME.PART`: part of a property of the object
          `OBJECTNAME`, where `PART` is `real` or `imag`

        - `.PROPERTYNAME` and `.PROPERTYNAME.PART`: property or part of a
          property of the parent object; the leading `.` avoids the
          ambiguity of `PROPERTYNAME.PART` (which is read as
          `OBJECTNAME.PROPERTYNAME`)

        - `TABLE:OBJECTNAME.PROPERTYNAME` and
          `TABLE:OBJECTNAME.PROPERTYNAME.PART`: property or part of a property
          of the object `OBJECTNAME` in another case data table (e.g.,
          `gen:2.PMAX` in a `branch` player)

        Object names are case row numbers (not bus ids) unless a mapping is
        given for their table (`mapping` for `name`, `mappings` for other
        tables). `OBJECTNAME` may itself contain `.`; the column name is
        split from the right. A `TABLE:` prefix is recognized only when
        `TABLE` is a table in the case; otherwise the `:` is part of
        `OBJECTNAME`. The `imag` part can only be set on complex data
        tables.

        # Arguments

        - `name`: data set name (e.g., bus, branch)

        - `file`: file name from which data is input

        - `parent`: name of the parent object; required only when a column
          does not specify `OBJECTNAME`

        - `scale`: scaling factor to apply to input data

        - `offset`: offset to apply to the scaled data

        - `mapping`: maps object names to rows, either as a dict of object
          names to row numbers, the name of a column of the `name` data whose
          values are the object names (e.g., `"BUS_I"`), or `(column,dtype)`
          to convert those values before matching (default dtype is `str`)

        - `mappings`: mappings of object names to rows for other tables,
          keyed by table name (e.g., `{"gen":("GEN_BUS",int)}`), in the same
          forms as `mapping`; `mapping` takes precedence for `name`

        # Exceptions

        - `AssertionError`: invalid table, file, column name, object name, or
          missing parent

        # See also

        - `pypower_sim.ppmodel.PPModel.players`

        - `pypower_sim.ppdata.PPData.set_input`
        """
        assert name in self.model.standard_idx, f"{name=} is not valid"
        assert name in self.model.case, f"{name=} is not found in the case"
        assert file not in self.model.players, f"player({file=}) already defined"
        assert os.path.exists(file), f"{file=} not found"
        data = pd.read_csv(file,index_col=[0],parse_dates=[0]) * scale + offset
        data.index.name = "datetime"

        mappings = dict(mappings) if mappings else {}
        for table in mappings:
            assert table in self.model.standard_idx and table in self.model.case, \
                f"mappings table {table!r} is not found in the case"
        if mapping is not None:
            mappings[name] = mapping
        parent_row = None if parent is None else \
            self._get_row(name,parent,mappings.get(name))
        targets = []
        for column in data.columns:
            table,rest = self._split_table(str(column),name)
            header = self.model.get_header(table)
            obj,prop,part = self._parse_player_column(rest,header)
            assert table == name or obj is not None, \
                f"player({file=}) column {column!r} must name an object of {table} data"
            assert prop in header and header.index(prop) < self.model.case[table].shape[1], \
                f"player({file=}) column {column!r} property {prop!r} is not found in {table} data"
            assert part != "imag" or np.iscomplexobj(self.model.case[table]), \
                f"player({file=}) column {column!r} cannot set the imaginary part " \
                f"of real {table} data"
            if obj is None:
                assert parent_row is not None, \
                    f"player({file=}) column {column!r} requires a parent object"
                row = parent_row
            else:
                row = self._get_row(table,obj,mappings.get(table))
            targets.append({
                "column": str(column),
                "name": table,
                "row": row,
                "property": prop,
                "part": part,
                })

        self.model.players[file] = {
            "name": name,
            "parent": parent_row,
            "data": data,
            "targets": targets,
            }

    def _split_table(self,column:str,name:str) -> tuple[str,str]:
        """@private Split an optional `TABLE:` prefix from a player column name"""
        if ":" in column:
            table,rest = column.split(":",1)
            if table in self.model.standard_idx and table in self.model.case:
                return table,rest
        return name,column

    @staticmethod
    def _parse_player_column(column:str,header:list[str]) -> tuple[str|None,str,str|None]:
        """@private Split a player column name into `(object,property,part)`"""
        if column.startswith("."):
            tokens = column[1:].split(".")
            assert 1 <= len(tokens) <= 2 and all(tokens), \
                f"player column {column!r} must be '.PROPERTYNAME' or '.PROPERTYNAME.PART'"
            part = tokens[1] if len(tokens) == 2 else None
            assert part in (None,)+PLAYER_PARTS, \
                f"player column {column!r} part {part!r} is not one of {PLAYER_PARTS}"
            return None,tokens[0],part
        tokens = column.split(".")
        if len(tokens) == 1:
            return None,column,None
        if len(tokens) >= 3 and tokens[-1] in PLAYER_PARTS and tokens[-2] in header:
            obj,prop,part = ".".join(tokens[:-2]),tokens[-2],tokens[-1]
        else:
            obj,prop,part = ".".join(tokens[:-1]),tokens[-1],None
        assert obj and prop, f"player column {column!r} is not valid"
        if prop in PLAYER_PARTS and obj in header:
            raise AssertionError(f"player column {column!r} is ambiguous; "
                f"use '.{column}' to refer to part {prop!r} of the parent's {obj!r}")
        return obj,prop,part

    def _get_row(self,name:str,obj:int|str,mapping:dict|str|tuple[str,type]|None) -> int:
        """@private Get the row number of object `obj` in the `name` data"""
        nrows = len(self.model.case[name])
        if mapping is None:
            try:
                row = int(obj)
            except ValueError as err:
                raise AssertionError(f"object {obj!r} is not a {name} row number") from err
        elif isinstance(mapping,dict):
            key = obj if obj in mapping else str(obj)
            assert key in mapping, f"object {obj!r} is not found in mapping"
            row = int(mapping[key])
        else:
            column,dtype = mapping if isinstance(mapping,tuple) else (mapping,str)
            target = self.model.get_data(name)
            assert column in target.columns, \
                f"{mapping=} is not a valid columns name in {name} data"
            rows = np.where(target[column].astype(dtype).values == dtype(obj))[0]
            assert len(rows) == 1, \
                f"object {obj!r} matches {len(rows)} rows of {name} data in column {column!r}"
            row = int(rows[0])
        assert 0 <= row < nrows, f"object {obj!r} row {row} is not in {name} data"
        return row

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
