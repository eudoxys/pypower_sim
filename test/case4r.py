from numpy import array
def case4r():
    return {
        "version": '2',
        "baseMVA": 100.0,
        "bus": array([
            # BUS_I, BUS_TYPE,  PD, PQ, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN 
            [     1,        3,   0,  0,  0,  0,        1,  1,  0,     230,    1,  1.1,  0.9],
            [     2,        1,   0,  0,  0,  0,        1,  1,  0,      69,    1,  1.1,  0.9],
            [     3,        1, 200, 10,  0,  0,        1,  1,  0,      69,    1,  1.1,  0.9],
            [     4,        2, 200, 10,  0,  0,        1,  1,  0,      69,    1,  1.1,  0.9],
            ]),
        "gen": array([
            # GEN_BUS, PG, QG, QMAX, QMIN,   VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN, QC1MAX, QC2MIN, QC1MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF
            [       1,  0,  0,  100, -100, 1.00,   100,          1,  400,    0,   0,   0,      0,      0,      0,      0,        0,       0,       0,      0,   0],
            # [       4,  0,  0,   20,  -20, 1.00,   100,          1,  150,    0,   0,   0,      0,      0,      0,      0,        0,       0,       0,      0,   0],
        ]),
        "branch": array([
            # F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX
            [     1,     2, 0.001, 0.02, 0.10,    400,    400,    500,   0,     0,         1,   -360,    360],
            [     2,     3, 0.002, 0.05, 0.08,    200,    200,    200,   0,     0,         1,   -360,    360],
            [     2,     4, 0.002, 0.05, 0.08,    200,    200,    200,   0,     0,         1,   -360,    360],
            [     3,     4, 0.003, 0.08, 0.12,     50,     50,     50,   0,     0,         1,   -360,    360],
            ]),
        "gencost": array([
            # MODEL, STARTUP, SHUTDOWN, N, COST0, ...
            [     2,     0.0,      0.0, 3,  0.04, 20.0, 0.0],
            # [     2,     0.0,      0.0, 3,  0.10, 50.0, 0.0],
            ]),
        }

if __name__ == '__main__':

    import numpy as np
    import pandas as pd
    import sys

    pd.options.display.max_columns = None
    pd.options.display.width = None
    osp_options = {
        "reference_voltage": complex(0.98,0),
        "cvx_solver":{"verbose":False},
        }

    case = case4r()

    from pypower_sim.ppmodel import PPModel, idx_gen
    model = PPModel(name="case4r",case=case)
    # model.options["VERBOSE"] = 3
    # model.options["OUT_ALL"] = 1

    from pypower_sim.ppsolver import PPSolver
    solver = PPSolver(model)

    print("",f"****{'*'*50}****",f"*** {'BEFORE OSP':^50s} ***",f"****{'*'*50}****","",sep="\n",flush=True)    
    model.print()

    try:
        assert solver.solve_opf(), "OPF failed"
        print("",f"****{'*'*50}****",f"*** {'INITIAL OPF':^50s} ***",f"****{'*'*50}****","",sep="\n",flush=True)
        model.print()
    except AssertionError as err:
        print("ERROR:",err,file=sys.stderr,flush=True)
    
    try:
        assert solver.solve_pf(), "PF failed"
        print("",f"****{'*'*50}****",f"*** {'INITIAL PF':^50s} ***",f"****{'*'*50}****","",sep="\n",flush=True)
        model.print()
    except AssertionError as err:
        print("ERROR:",err,file=sys.stderr,flush=True)

    violations = model.get_violations()
    if violations:
        print("")
        print("*** VIOLATIONS ***",*violations,"",sep="\n  ")

        print("","*** SOLVING OSP ***","",sep="\n")
        status,result = solver.solve_osp(options=osp_options,with_result=True)
        if status:

            for u,v in {"Generators":"MW","Capacitors":"MVAr","Condensers":"MVAr"}.items():
                print("",f"New {u}","--------------------",*[f"  {n:4d}. {x:8.3f} {v}" for n,x in enumerate(result[u.lower()].tolist())],sep="\n  ")
                print("  --------------------")
                print(f"    Total {round(sum(result[u.lower()]),2):8.3f} {v}")

            print("",f"Bus Voltages","--------------------",*[f"  {n:4d}. {np.abs(x):8.3f} pu.kV @ {np.angle(x)*180/np.pi:8.3f} deg" for n,x in enumerate(result["voltages"].tolist())],sep="\n  ")
            print("  --------------------")

            print("",f"****{'*'*50}****",f"*** {'AFTER OSP':^50s} ***",f"****{'*'*50}****","",sep="\n",flush=True)
            model.print()
            try:
                assert solver.solve_opf(), "OPF failed"
                print("",f"****{'*'*50}****",f"*** {'AFTER OPF':^50s} ***",f"****{'*'*50}****","",sep="\n",flush=True)
                model.print()
            except AssertionError as err:
                print("ERROR:",err,file=sys.stderr,flush=True)

            # set bus voltages according to OSP results instead of OPF results for testing purposes
            # model.case["gen"][:,idx_gen.VG] = result["voltages"][model._bus_i(model.case["gen"][:,idx_gen.GEN_BUS].astype(int).tolist())]
            
            try:
                assert solver.solve_pf(), "PF failed"
                print("",f"****{'*'*50}****",f"*** {'AFTER PF':^50s} ***",f"****{'*'*50}****","",sep="\n",flush=True)
                model.print()
            except AssertionError as err:
                print("ERROR:",err,file=sys.stderr,flush=True)

            violations = model.get_violations()
            if violations:
                print("")
                print("*** VIOLATIONS ***",*violations,"",sep="\n  ")

        else:

            print("OSP solution failed")
