"""Test script"""

import sys
import datetime as dt
import pytz
from pypower_sim import PPModel
from pypower_sim import PPSolver
from pypower_sim import PPPlots
from pypower_sim import PPData
from pypower_sim import PPGIS
import _cases as cases

n_tests = 0
n_failed = 0

DEBUG = "--debug" in sys.argv or __name__ == "__main__"

start = dt.datetime(2020,7,31,17,0,0,0,pytz.UTC)
end = dt.datetime(2020,8,1,16,0,0,0,pytz.UTC)

for test in [x for x in dir(cases) if x.startswith("test_")]:

    print(f"Running {test} from {start} to {end}...",flush=True)

    n_tests += 1

    try:
        
        test_model = PPModel(name="test_model",case=getattr(cases,test))        
        gis = PPGIS(test_model,"gis.csv") if "gis" in test_model.case else None
        solver=PPSolver(test_model)
        plots = PPPlots(test_model)

        print("  Checking violations...",len(test_model.get_violations()),"found",flush=True)

        print("  Generating plots",end="...",flush=True)
        plots.voltage().savefig(f"{test}_voltage_original.png")
        print("ok")

        print("  Running initial AC OPF...","ok" if solver.solve_opf(use_acopf=True) else "AC OPF failed",flush=True)
        print("  Running initial powerflow...","ok" if solver.solve_pf() else "Powerflow failed",flush=True)
        print("  Checking violations...",len(test_model.get_violations()),"found",flush=True)

        if test_model.get_violations():
            print("  Running OSP...","ok" if solver.solve_osp() else "Powerflow failed",flush=True)
            print("  Checking AC OPF...","ok" if solver.solve_opf(use_acopf=True) else "AC OPF failed",flush=True)
            print("  Checking powerflow...","ok" if solver.solve_pf() else "Powerflow failed",flush=True)
            print("  Checking violations...",len(test_model.get_violations()),"found",flush=True)

        print("  Generating plots",end="...",flush=True)
        plots.voltage().savefig(f"{test}_voltage_initial.png")
        print("ok")

        print("  Running JSON round robin",end="...",flush=True)
        test_model.save(f"{test}_model.json",indent=1)
        test_model.load(f"{test}_model.json")
        print("ok")

        print("  Loading timeseries inputs",end="...",flush=True)
        tapes = PPData(test_model)
        tapes.set_input("bus","PD",f"load.csv",scale=10)
        tapes.set_input("bus","QD",f"load.csv",scale=1)
        print("ok")

        print("  Configuring timeseries outputs",end="...",flush=True)
        tapes.set_output("bus","VM",f"{test}_bus_vm.csv",formatting=".3f")
        tapes.set_output("bus","VA",f"{test}_bus_va.csv",formatting=".4f")
        tapes.set_output("bus","PD",f"{test}_bus_pd.csv",formatting=".4f")
        tapes.set_output("bus","QD",f"{test}_bus_qd.csv",formatting=".4f")
        tapes.set_recorder(f"{test}_cost.csv","cost",["cost"],
            scale=test_model.case['baseMVA'],formatting=".2f")
        print("ok")

        test_solver = PPSolver(test_model)
        print("  Running timeseries simulation",end="...",flush=True)
        print("  ok" if test_solver.run_timeseries(
            start=start,
            end=end,
            freq="1h",
            progress=lambda **kwargs: print(end=".",flush=True),
            ) is None else "original timeseries simulation failed",flush=True)

        print("  Generating plots",end="...",flush=True)
        plots.voltage().savefig(f"{test}_voltage_final.png")
        print("ok")

        print("  Running JSON round robin",end="...",flush=True)
        test_model.save(f"{test}_model.json",indent=1)
        test_model.load(f"{test}_model.json")
        print("ok" if solver.solve_pf() else "powerflow failed after load()",flush=True)

        if "gis" in test_model.case:
            print("  Saving GIS data",end="...",flush=True)
            test_model.get_data("gis").to_csv(f"{test}_gis.csv",index=False,header=True)
            print("ok",flush=True)

        print(f"  {test} OK")

    except Exception as err:
        
        if DEBUG:
            raise

        print(f"  {test} FAILED:",err)
        n_failed += 1

print(n_tests,"completed.",n_failed,"failures")
exit(n_failed)

