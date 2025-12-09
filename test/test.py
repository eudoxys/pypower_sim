"""Test script"""

import sys
import datetime as dt
import pytz
from pypower_sim import PPModel
from pypower_sim import PPSolver
from pypower_sim import PPPlots
from pypower_sim import PPData
from pypower_sim import PPGIS
import cases

n_tests = 0
n_failed = 0

DEBUG = "--debug" in sys.argv or __name__ == "__main__"

for test in [x for x in dir(cases) if x.startswith("test_")]:

    print(f"Running {test}",end="...",flush=True)
    n_tests += 1

    try:
        
        test_model = PPModel(name="test_model",case=getattr(cases,test))
        
        gis = PPGIS(test_model,"gis.csv")

        solver=PPSolver(test_model)
        assert solver.solve_opf(), "DC OPF failed"
        assert solver.solve_opf(use_acopf=True), "AC OPF failed"
        assert solver.solve_pf(), "Powerflow failed"

        plots = PPPlots(test_model)
        plots.voltage().savefig(f"{test}_voltage.png")

        tapes = PPData(test_model)
        tapes.set_input("bus","PD",f"load.csv",scale=10)
        tapes.set_input("bus","QD",f"load.csv",scale=1)
        tapes.set_output("bus","VM",f"{test}_bus_vm.csv",formatting=".3f")
        tapes.set_output("bus","VA",f"{test}_bus_va.csv",formatting=".4f")
        tapes.set_output("bus","PD",f"{test}_bus_pd.csv",formatting=".4f")
        tapes.set_output("bus","QD",f"{test}_bus_qd.csv",formatting=".4f")
        tapes.set_recorder(f"{test}_cost.csv","cost",["cost"],
            scale=test_model.case['baseMVA'],formatting=".2f")

        test_solver = PPSolver(test_model)
        start = dt.datetime(2020,7,31,17,0,0,0,pytz.UTC)
        end = dt.datetime(2020,8,1,16,0,0,0,pytz.UTC)
        assert test_solver.run_timeseries(
            start=start,
            end=end,
            freq="1h",
            ) is None, "original timeseries simulation failed"

        assert solver.solve_pf(), "Powerflow failed after load()"

        test_model.save("test_model.json",indent=1)
        test_model.load("test_model.json")
        assert test_solver.run_timeseries(
            start=start,
            end=end,
            freq="1h",
            ) is None, "loaded timeseries simulation failed"

        print("OK")

    except Exception as err:
        
        if DEBUG:
            raise

        print("FAILED:",err)
        n_failed += 1

print(n_tests,"completed.",n_failed,"failures")
exit(n_failed)

