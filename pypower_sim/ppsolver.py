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

import numpy as np
import pandas as pd

from pypower.runpf import runpf
from pypower.rundcopf import rundcopf
from pypower.runopf import runopf as runacopf
from pypower.ppoption import ppoption
from pypower import idx_gen, idx_bus
from pypower_sim.runosp import runosp

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
        use_acopf:bool=True,
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
        options:dict[str:str|int|float|dict]|None=None,
        update:str='success',
        with_result:bool=False,
        costs:dict[str,float]|None=None,
        generators:dict[str,float]={
            "gen":{
                "GEN_STATUS": 1,
                "MBASE": 100,
                },
            "gencost":{
                "MODEL": 2, # polynomial function
                "NCOST": 2, # linear cost
                "COST0": 50, # operating cost ($/MWh)
                "COST1": 0, # fixed cost ($/h)
                },
            "roundup":-1,
            },
        capacitors:dict[str,float]={
            "gen":{
                "GEN_STATUS": 1,
                "MBASE": 100,
                },
            "gencost":{
                "MODEL": 2, # polynomial function
                "NCOST": 2, # linear cost
                "COST0": 0, # operating cost ($/MWh)
                "COST1": 0, # fixed cost ($/h)
                },
            "roundup":0,
            },
        condensers:dict[str,float]={
            "gen":{
                "GEN_STATUS": 1,
                "MBASE": 100,
                },
            "gencost":{
                "MODEL": 2, # polynomial function
                "NCOST": 2, # linear cost
                "COST0": 0, # operating cost ($/MWh)
                "COST1": 0, # fixed cost ($/h)
                },
            "roundup":0,
            },
        ):
        """Solve the optimal sizing placement problem

        # Arguments

        - `options`: specify problem options to enable/disable
          (see `pypower_sim.runosp.OspConfig`)

        - `update`: when to update of model case data (must be in `
          {'always', 'success', 'failure', 'never'}`)
    
        - `with_result`: include result in return value

        - `costs`: specify capacity expansion costs

        - `generators`: default `gen`, `gencost`, and `roundup` values for new
          generators

        - `capacitors`: default `gen`, `gencost`, and `roundup` values for new
          generators

        - `condensers`: default `gen`, `gencost`, and `roundup` values for new
          generators

        # Returns

        - `bool`: `True` on success, `False` on failure

        - `dict`: result (if `with_result` is `True`)

        # Description

        The following `costs` may specified:

        1. `capacitor`: specifies the capacitor addition cost in $/MVAr
        (default `100`)

        2. `condenser`: specifies the condenser addition cost in $/MVAr
        (default `1000`)

        3. `generation`: specifies the generation addition cost in $/MVA
        (default `1000`)

        4. `curtailment`: specifies the load curtailment cost in $/MW
        (default `10000`)

        Note that in general the costs should increase in the order presented
        above, i.e., capacitors are the least costly and load curtailment is
        the most costly.

        # Caveat

        The optimal sizing/placement solver is experimental.
        """
        # warnings.warn("solver_osp is not implemented yet")
        result = runosp(self.model,config=options)
        status = result["status"]
        success = status == 1
        if ( success and update in ["always","success"] ) \
                or ( not success and update in ["always","failure"] ):

            # update gen and gencost data
            bus = self.model.get_data("bus")
            gen = self.model.get_data("gen")
            gencost = self.model.get_data("gencost")

            # add new generators to gen busses
            if sum(result["generators"]) > 0:
                gen_bus, pmax = np.array([(int(x),y) for x,y in enumerate(result["generators"]) if round(y,3) > 0]).T
                if "roundup" in generators:
                    pmax = np.ceil(pmax*10**generators["roundup"])/10**generators["roundup"]
                newgen = {
                    "GEN_BUS":bus.iloc[gen_bus].BUS_I,
                    "PMAX":pmax,
                    "VG": np.abs(result["voltages"][gen_bus.astype(int)]).round(3)
                    }
                newgen.update({x:[y]*(len(gen_bus)) for x,y in generators["gen"].items()})
                newgencost = {x:[y]*(len(gen_bus)) for x,y in generators["gencost"].items()}
            else:
                newgen = {x:[] for x in gen.columns}
                newgencost = {x:[] for x in gencost.columns}
            # TODO: remove PQ bus entries

            # add active capacitors to gen busses
            if sum(result["capacitors"]) > 0:
                cap_bus, qmax = np.array([(x,y) for x,y in enumerate(result["capacitors"]) if round(y,3) > 0]).T
                if "roundup" in capacitors:
                    qmax = np.ceil(qmax*10**capacitors["roundup"])/10**capacitors["roundup"]
                newcap = {
                    "GEN_BUS": bus.iloc[cap_bus].BUS_I,
                    "QMAX":qmax,
                    "VG": np.abs(result["voltages"][cap_bus.astype(int)]).round(3)
                    }
                newcap.update({x:[y]*(len(cap_bus)) for x,y in capacitors["gen"].items()})
                newcapcost = {x:[y]*(len(cap_bus)) for x,y in capacitors["gencost"].items()}
            else:
                newcap = {x:[] for x in gen.columns}
                newcapcost = {x:[] for x in gencost.columns}
            # TODO: remove PQ bus entries

            # add active condensers to gen busses
            if sum(result["condensers"]) > 0:
                con_bus, qmin = np.array([(x,y) for x,y in enumerate(-result["condensers"]) if round(y,3) > 0]).T
                if "roundup" in condensers:
                    qmin = np.ceil(qmin*10**condensers["roundup"])/10**condensers["roundup"]
                newcon = {
                    "GEN_BUS": bus.iloc[con_bus].BUS_I,
                    "QMIN": qmin,
                    "VG": np.abs(result["voltages"][cons_bus.astype(int)]).round(3)
                    }.update(condensers["gen"])
                newcon.update({x:[y]*(len(con_bus)) for x,y in condensers["gen"].items()})
                newcapcost = {x:[y]*(len(con_bus)) for x,y in condensers["gencost"].items()}
            else:
                newcon = {x:[] for x in gen.columns}
                newconcost = {x:[] for x in gencost.columns}
            # TODO: remove PQ bus entries

            # add passive capacitors to PQ busses
            # TODO

            # add passive condensers to PQ busses
            # TODO

            # compile new gen array
            gen = pd.concat([gen,
                pd.DataFrame(newgen),
                pd.DataFrame(newcap),
                pd.DataFrame(newcon),
                ]).fillna(0).reset_index(drop=True)
            self.model.case["gen"] = gen.values

            # upgrade PQ busses that now have gens
            # bustype2 = set(x for x in self.model._bus_i(gen.GEN_BUS.values.astype(int).tolist()) if bus.iloc[x].BUS_TYPE == 1)
            # self.model.case["bus"][list(bustype2),idx_bus.BUS_TYPE] = 2

            # compile new gencost array
            if len(gencost) > 0:
                gencost = pd.concat([gencost,
                    pd.DataFrame(newgencost),
                    pd.DataFrame(newcapcost),
                    pd.DataFrame(newconcost),
                    ]).fillna(0).reset_index(drop=True)
                self.model.case["gencost"] = gencost.values

        if with_result:
            return status,result
        else:
            return status

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
        use_acopf:bool=True,
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

    show_violations = False
    show_totals = True

    def show_results(model):
        violations = model.get_violations()
        if violations or show_totals:
            generators = complex(*model.get_data("gen")[["PG","QG"]].values.sum(axis=0))
            loads = complex(*model.get_data("bus")[["PD","QD"]].values.sum(axis=0))
            losses = generators - loads
            print(f"{generators=:.1f}, {generators=:.1f}, {losses=:.1f}, violations={len(violations)}")
            if show_violations:
                print(*violations,sep="\n")
        else:
            print("OK")

    try:
        import wecc240
    except ModuleNotFoundError as err:
        import os
        if str(err) == "No module named 'wecc240'" and os.system("pip install git+https://github.com/eudoxys/wecc240") == 0:
            import wecc240
        else:
            raise

    try:
        from pypower_sim.ppmodel import PPModel
    except ModuleNotFoundError as err:
        import os
        if str(err) == "No module named 'pypower_sim'" and os.system("pip install -e ..") == 0:
            from pypower_sim.ppmodel import PPModel
        else:
            raise

    for test in [x for x in dir(wecc240) if x.startswith("wecc240_")]:

        print(f"Testing {test}...")
        module = getattr(wecc240,test)
        model = PPModel(case=getattr(module,test)())

        solver = PPSolver(model)

        for problem,method in {
            "original model powerflow": solver.solve_pf,
            "original model OPF": solver.solve_opf,
            "original model OSP": solver.solve_osp,
            "optimal model OPF": solver.solve_opf,
            "optimal model PF": solver.solve_pf,
            }.items():
            print(f"Solving {problem}",end="...",flush=True)
            if method():
                show_results(model)
            else:
                print("ERROR:",method.__name__,"failed")
                solver.model.options["VERBOSE"] = 3
                solver.model.options["OUT_ALL"] = 1
                method()
                solver.model.options = solver.model.default_options

        # print("Solving original model powerflow",end="...",flush=True)
        # if solver.solve_pf():
        #     show_results(model)
        # else:
        #     print("PF failed")

        # print("Solving original model OPF",end="...",flush=True)
        # if solver.solve_opf(use_acopf=True):
        #     show_results(model)
        # else:
        #     print("OPF failed")
        #     solver.model.options["VERBOSE"] = 3
        #     solver.model.options["OUT_ALL"] = 1
        #     solver.solve_opf(use_acopf=True)
        #     solver.model.options = solver.model.default_options

        # print("Solving original OPF model powerflow",end="...",flush=True)
        # if solver.solve_pf():
        #     show_results(model)
        # else:
        #     print("PF failed")
        #     solver.model.options["VERBOSE"] = 3
        #     solver.model.options["OUT_ALL"] = 1
        #     solver.solve_pf()
        #     solver.model.options = solver.model.default_options

        # print("Solving original OSP model",end="...",flush=True)
        # if solver.solve_osp():
        #     show_results(model)
        # else:
        #     print("OSP failed")


        # print("Solving optimal model OPF",end="...",flush=True)
        # if solver.solve_opf(use_acopf=True):
        #     show_results(model)
        # else:
        #     print("OPF failed")
        #     solver.model.options["VERBOSE"] = 3
        #     solver.model.options["OUT_ALL"] = 1
        #     solver.solve_opf(use_acopf=True)
        #     solver.model.options = solver.model.default_options

        # print("Solving optimal OPF model powerflow",end="...",flush=True)
        # if solver.solve_pf():
        #     show_results(model)
        # else:
        #     print("PF failed")
        #     solver.model.options["VERBOSE"] = 3
        #     solver.model.options["OUT_ALL"] = 1
        #     solver.solve_pf()
        #     solver.model.options = solver.model.default_options
