# script to check a case
def check_case(name,case):

    print(f"Running {name}...")
    print_model = False

    import numpy as np
    import pandas as pd
    import sys

    pd.options.display.max_columns = None
    pd.options.display.width = None
    osp_options = {
        "reference_voltage": complex(0.98,0),
        "cvx_solver":{"verbose":False},
        }

    from pypower_sim.ppmodel import PPModel, idx_gen
    model = PPModel(name=name,case=case)
    # model.options["VERBOSE"] = 3
    # model.options["OUT_ALL"] = 1

    from pypower_sim.ppsolver import PPSolver
    solver = PPSolver(model)

    if print_model:
        print("",f"****{'*'*50}****",f"*** {'BEFORE OSP':^50s} ***",f"****{'*'*50}****","",sep="\n",flush=True)    
        model.print()

    try:
        assert solver.solve_pf(), "Initial PF failed"
        if print_model:
            print("",f"****{'*'*50}****",f"*** {'INITIAL PF':^50s} ***",f"****{'*'*50}****","",sep="\n",flush=True)
            model.print()
        else:
            print("Initial PF ok",flush=True)
    except AssertionError as err:
        print("ERROR:",err,file=sys.stderr,flush=True)

    try:
        assert solver.solve_opf(), "Initial OPF failed"
        if print_model:
            print("",f"****{'*'*50}****",f"*** {'INITIAL OPF':^50s} ***",f"****{'*'*50}****","",sep="\n",flush=True)
            model.print()
        else:
            print("Initial OPF ok",flush=True)
    except AssertionError as err:
        print("ERROR:",err,file=sys.stderr,flush=True)

    try:
        assert solver.solve_pf(), "Optimal PF failed"
        if print_model:
            print("",f"****{'*'*50}****",f"*** {'INITIAL PF':^50s} ***",f"****{'*'*50}****","",sep="\n",flush=True)
            model.print()
        else:
            print("Optimal PF ok",flush=True)

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

            if print_model:
                print("",f"****{'*'*50}****",f"*** {'AFTER OSP':^50s} ***",f"****{'*'*50}****","",sep="\n",flush=True)
                model.print()

            try:
                assert solver.solve_pf(), "Sized PF failed"
                if print_model:
                    print("",f"****{'*'*50}****",f"*** {'AFTER PF':^50s} ***",f"****{'*'*50}****","",sep="\n",flush=True)
                    model.print()
                else:
                    print("Resized PF ok",flush=True)
                violations = model.get_violations()
                if violations:
                    print("")
                    print("*** VIOLATIONS ***",*violations,"",sep="\n  ")
            except AssertionError as err:
                print("ERROR:",err,file=sys.stderr,flush=True)


            try:
                assert solver.solve_opf(), "Sized OPF failed"
                if print_model:
                    print("",f"****{'*'*50}****",f"*** {'AFTER OPF':^50s} ***",f"****{'*'*50}****","",sep="\n",flush=True)
                    model.print()
                else:
                    print("Sized OPF ok",flush=True)
            except AssertionError as err:
                print("ERROR:",err,file=sys.stderr,flush=True)

            # set bus voltages according to OSP results instead of OPF results for testing purposes
            # model.case["gen"][:,idx_gen.VG] = result["voltages"][model._bus_i(model.case["gen"][:,idx_gen.GEN_BUS].astype(int).tolist())]
            
            try:
                assert solver.solve_pf(), "Reoptimized PF failed"
                if print_model:
                    print("",f"****{'*'*50}****",f"*** {'AFTER PF':^50s} ***",f"****{'*'*50}****","",sep="\n",flush=True)
                    model.print()
                else:
                    print("Reoptimized PF ok",flush=True)
                violations = model.get_violations()
                if violations:
                    print("")
                    print("*** VIOLATIONS ***",*violations,"",sep="\n  ")
            except AssertionError as err:
                print("ERROR:",err,file=sys.stderr,flush=True)


        else:

            print("OSP solution failed")
