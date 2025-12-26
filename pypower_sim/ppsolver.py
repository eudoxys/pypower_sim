"""PyPOWER solver

This model implements the PyPOWER steady-state powerflow (PF), optimal
powerflow (OPF), and quasi-steady time series (QSTS) solvers.  

# Description 

The following solvers are available:

1. Powerflow (PF)

    `pypower_sim.ppsolver.PPSolver.solve_pf`: solves steady-state powerflow
    (PF) problem

2. Optimal Powerflow (OPF)

    `pypower_sim.ppsolver.PPSolver.solve_opf`: solve optimal powerflow problem
    (DC or AC OPF).

3. Optimal Sizing and Placement (OSP)

    `pypower_sim.ppsolver.PPSolver.solve_osp`: solve optimal sizing/placement
    problem (OSP).

4. Time-series Simulation (QSTS)

    `pypower_sim.ppsolver.PPSolver.run_timeseries`: solver quasi-steady
    time-series (QSTS) problems.

# Example

This example loads the WECC 240 case and solve an hourly timeseries simulation
for August 2020.

    from wecc240 import wecc240
    model = PPModel(case=wecc240)

    solver = PPSolver(model)
    solver.run_timeseries(
        start="2020-08-01 00:00:00+07:00",
        end="2020-09-01 00:00:00+07:00",
        freq="1h",
        )
"""

import datetime as dt
from time import time
from typing import Callable
import warnings

import pandas as pd

from pypower.runpf import runpf
from pypower.rundcopf import rundcopf
from pypower.runopf import runopf as runacopf
from pypower.ppoption import ppoption
from .runosp import runosp

class PPSolver:
    """PyPOWER solver implementation"""
    def __init__(self,model):
        """PyPOWER model solver"""
        self.model = model

    def solve_pf(self,
        update:str='success',
        with_result:bool=False,
        ) -> [bool,dict]:
        """Solve the powerflow problem

        # Arguments

        - `update`: when to update of model case data (must be in `
          {'always', 'success', 'failure', 'never'}`)

        - `with_result`: include result in return value

        # Returns

        - `bool`: `True` on success, `False` on failure

        - `dict`: result (if `with_result` is `True`)
        """
        assert update in ["always","success","failure","never"], f"{update=} is invalid"
        result,status = runpf(self.model.case,ppoption(**self.model.options))
        success = status == 1
        if ( success and update in ["always","success"] ) \
                or ( not success and update in ["always","failure"] ):
            for name,values in result.items():
                if name in self.model.case:
                    self.model.case[name] = values
        if with_result:
            return status==1,result
        return status==1

    def solve_opf(self,
        use_acopf:bool=False,
        update:str='success',
        with_result:bool=False,
        ) -> [bool,dict]:
        """Solve the optimal powerflow problem

        # Arguments

        - `use_acopf`: enable AC OPF solution

        - `update`: when to update of model case data (must be in `
          {'always', 'success', 'failure', 'never'}`)
    
        - `with_result`: include result in return value

        # Returns

        - `bool`: `True` on success, `False` on failure

        - `dict`: result (if `with_result` is `True`)
        """
        assert use_acopf in [True,False], f"{use_acopf=} is invalid"
        assert update in ["always","success","failure","never"], f"{update=} is invalid"
        opf = (runacopf if use_acopf else rundcopf)
        result = opf(self.model.case,ppoption(**self.model.options))
        success = result["success"] is True
        if ( success and update in ["always","success"] ) \
                or ( not success and update in ["always","failure"] ):
            for name,values in result.items():
                if name in self.model.case:
                    self.model.case[name] = values
            self.model.case["cost"] = result["f"]
        if with_result:
            return success,result
        return success

    def solve_osp(self,
        costs:dict[str:float]|None=None,
        options:dict[str:str|int|float]|None=None,
        update:str='success',
        with_result:bool=False):
        """Solve the optimal sizing placement problem

        # Arguments

        - `costs`: specify capacity expansion costs

        - `options`: specify problem options to enable/disable

        - `update`: when to update of model case data (must be in `
          {'always', 'success', 'failure', 'never'}`)
    
        - `with_result`: include result in return value

        # Returns

        - `bool`: `True` on success, `False` on failure

        - `dict`: result (if `with_result` is `True`)

        # Description

        The following `costs` may specified:

        1. `capacitor`: specifies the capacitor addition cost in $/MVAr (default `100`)

        2. `condenser`: specifies the condenser addition cost in $/MVAr (default `1000`)

        3. `generation`: specifies the generation addition cost in $/MVA (default `1000`)

        4. `curtailment`: specifies the load curtailment cost in $/MW (default `10000`)

        Note that in general the costs should increase in the order presented
        above, i.e., capacitors are the least costly and load curtailment is
        the most costly.

        The following `options` are supported:

        - `verbose`: enable verbose output from `cvxpy` (default `False`)

        - `show_data`: output problem data before solving (default `False`)

        - `complex`: enable complex flows in optimal solution (default `False`)

        - `iterations`: maximum iterations allowed (default `10000`)

        - `runtime`: maximum runtime allowed (default `None`)

        - `angle`: voltage angle accuracy limit (default `10` degrees)

        - `magnitude`: voltage magnitude constraint (default `5` %)

        - `margin`: load capacity margin (default `20` %)

        # Caveat

        This solver is experimental and is currently being developed
        """
        warnings.warn("solver_osp is not implemented yet")

    def update_inputs(self,t:dt.datetime) -> int:
        """Synchronize inputs with the current date/time

        # Arguments

        - `t`: the current date/time
        """
        # update inputs
        errors = 0
        for name,spec in self.model.inputs.items():
            data = spec["data"]
            name,column = name
            column_number = self.model.get_header(name).index(column)
            mapping = spec["mapping"]["index"]
            scales = spec["mapping"]["scale"]
            try:
                target = self.model.case[name]
                target[mapping,column_number] = data.loc[t] * scales
            except KeyError as exception:
                warnings.warn(f"input({name=},{column=}) {exception=}")
                errors += 1
        return errors

    def update_outputs(self,
        t:dt.datetime,
        ts_format:str="%Y-%m-%d %H:%M:%S %Z"
        ) -> int:
        """Synchronize outputs to the current date/time

        # Arguments

        - `t`: the current date/time

        - `ts_format`: timestamp format
        """
        ts = t.strftime(ts_format)

        # output
        for file,spec in self.model.outputs.items():
            mapping = spec["mapping"]
            scale = mapping["scale"]
            offset = mapping["offset"]
            data = [f"{{0:{spec['format']}}}".format(x)
                for x in self.model.get_data(
                    spec["name"]).loc[mapping["rows"],spec["column"]]*scale + offset]
            print(ts,*data,sep=",",file=spec["fh"],flush=True)

        # recorders
        errors = 0
        for file,recorder in self.model.recorders.items():
            values = [ts]
            for _,spec in recorder["targets"].items():
                value = self.model.case
                for level in spec["source"]:
                    if not level in value:
                        warnings.warn(f"recorder '{file}' -- '{level}' not found")
                        errors += 1
                        break
                    value = value[level]
                if not isinstance(value,(int,float,bool,str,type(None))):
                    value = float('nan')
                elif isinstance(value,(int,float)):
                    value = eval(spec["transform"])(value)
                values.append(f"{{0:{spec['format']}}}".format(value))
            print(*values,sep=",",file=recorder["fh"],flush=True)
        return errors

    def run_timeseries(self,*args,
        # pylint: disable=too-many-arguments,too-many-locals
        progress:Callable=None,
        call_on_fail:Callable=None,
        stop_on_fail:bool=True,
        stop_test:Callable=None,
        use_acopf:bool=False,
        **kwargs) -> str|list[str]|None:
        """Run a timeseries simulation

        Time-series solutions perform the following sequence of operations at each
        timestep.

        ```mermaid
        sequenceDiagram

            autonumber
            note over inputs, players: Read data

            note over outputs, recorders: Open files

            loop from start to end by timestep

                inputs ->>model:Input data

                players ->>model:Player data

                model ->>+solvers:Initial/last result
                note over solvers: Call runopf()
                solvers ->>-model:OPF result

                model ->>+solvers:OPF result
                note over solvers: Call runpf()
                solvers ->>-model:PF result

                note over model: Save errors

                model ->>+outputs:Output data

                model ->>+recorders:Recorder data
            
            end

            note over model: Generate profile
        ```

        # Arguments

        - `*args`: See `pandas.date_range(*args)`

        - `progress`: set a progress callback function

        - `call_on_fail`: set a call-on-fail function

        - `stop_on_fail`: enable stop-on-fail condition

        - `stop_test`: set a stop test call back function

        - `use_acopf`: enable use of AC OPF instead of DC OPF

        - `**kwargs`: See `pandas.date_range(**kwargs)`
        
        # Returns

        - `None`: No errors to report

        - `str`: Error message (when stop_on_fail is True)

        - `list[str]`: Error messages (when stop_on_fail is False)
        """

        assert progress is None or callable(progress), \
            "progress must be callable or None"
        assert call_on_fail is None or callable(call_on_fail), \
            "call_on_fail must be callable or None"
        assert stop_test is None or callable(stop_test), \
            "stop_test must be callable or None"
        assert isinstance(stop_on_fail,bool), \
            "stop_on_fail must be boolean"
        assert isinstance(use_acopf,bool), \
            "use_acopf must be boolean"

        self.model.errors = [] # collect errors, if any
        tic0 = time()

        # process time specified range
        trange = pd.date_range(*args,**kwargs)
        niters = 0
        topf = 0.0
        tpf = 0.0

        # start recorders
        for file,recorder in self.model.recorders.items():
            recorder["fh"] = open(file,"w",encoding="utf-8")
            columns = ["timestamp"] + list(recorder["targets"].keys())
            print(*columns,sep=",",file=recorder["fh"],flush=True)

        # start outputs
        for file,output in self.model.outputs.items():
            output["fh"] = open(file,"w",encoding="utf-8")
            columns = ["timestamp"] + output["mapping"]["columns"]
            print(*columns,sep=",",file=output["fh"],flush=True)

        for t in (x.tz_convert("UTC") for x in trange):

            # setup time and progress/stop callback
            ts = t.strftime("%Y-%m-%d %H:%M:%S %Z")
            if callable(progress) and progress(f"""{ts} ({len(self.model.errors)
                    if self.model.errors else 'no'} errors)"""):
                return None

            # update inputs
            if self.update_inputs(t) > 0:
                if call_on_fail:
                    call_on_fail("Input error")
                if stop_on_fail:
                    break

            # solve OPF and check result
            status,result = self.solve_opf(use_acopf,with_result=True)
            if status is not True:
                failed = f"OPF failed at {ts}"
                self.model.errors.append(failed)
                if call_on_fail:
                    call_on_fail(failed)
                if stop_on_fail:
                    break
            topf += result["et"]

            # solver powerflow and check result
            status,result = self.solve_pf(with_result=True)
            if status is not True:
                failed = f"PF failed at {ts}"
                self.model.errors.append(failed)
                if call_on_fail:
                    call_on_fail(failed)
                if stop_on_fail:
                    break
            tpf += result["et"]

            # process outputs
            if self.update_outputs(t) > 0:
                if call_on_fail:
                    call_on_fail("Output error")
                if stop_on_fail:
                    break

            # check stop condition
            niters += 1
            if stop_test and stop_test(t):
                self.model.errors.append(f"Stopped at {t=}")
                break

        ttot = time() - tic0
        self.model.profile = {
            "Expected iterations": len(trange),
            "Completed iterations": niters,
            "Total OPF time (s)": round(topf,4),
            "Fraction OPF time (s/s)": round(topf/ttot,2) if ttot > 0 else 0,
            "Total PF time (s)": round(tpf,4),
            "Fraction PF time (s/s)": round(tpf/ttot,2) if ttot > 0 else 0,
            "Total run time (s)": round(ttot,4),
            "Iteration time (s/iter)": round(ttot/niters,4) if niters > 0 else "N/A",
        }

        return self.model.errors if self.model.errors else None

if __name__ == "__main__":

    # pip install git+https://github.com/eudoxys/wecc240
    from wecc240.wecc240_2011 import wecc240_2011 as wecc240

    # pip install -e ..
    from pypower_sim.ppmodel import PPModel
    model = PPModel(case=wecc240)

    solver = PPSolver(model)

    solver.solve_osp()
    solver.solve_opf()
    solver.solve_pf()
