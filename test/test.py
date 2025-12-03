"""Test script"""

from pypower_sim import PPModel
import cases

n_tests = 0
n_failed = 0
for test in [x for x in dir(cases) if x.startswith("test_")]:

    print(f"Run {test}",end="...",flush=True)
    n_tests += 1
    try:
        model = PPModel(case=getattr(cases,test))
        from pypower_sim import PPSolver
        solver=PPSolver(model)
        solver.solve_opf()
        solver.solve_opf(use_acopf=True)
        solver.solve_pf(with_result=True)
        print("OK")
    except Exception as err:
        print("FAILED:",err)
        n_failed += 1

print(n_tests,"completed.",n_failed,"failures")
exit(n_failed)

