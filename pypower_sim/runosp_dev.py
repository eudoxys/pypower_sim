import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _(mo):
    from runosp import __doc__ as docs
    _docs = docs.split("\n",1)
    mo.accordion({f"# {_docs[0]}":mo.md(_docs[1])})
    return


@app.cell
def _(mo):
    file_ui = mo.ui.file_browser(
        initial_path="../test",
        filetypes=[".py"],
        label="Choose as [`pypower_sim`](https://www.eudoxys.com/pypower_sim/) case file (*.py):",
        multiple=False,
        ignore_empty_dirs=True,
    )
    mo.accordion({"## Model selection":file_ui})
    return (file_ui,)


@app.cell
def _(mo, model):
    _data = {
        x: model.get_data(x)
        for x in ["bus", "branch", "gen", "gencost", "dcline", "dclinecost"]
        if x in model.case
    }
    mo.accordion(
        {
            "## Model Data": mo.ui.tabs(
                {
                    x.title(): mo.ui.table(
                        y,
                        selection=None,
                        text_justify_columns={z: "right" for z in y.columns},
                    )
                    for x, y in _data.items()
                }
            )
        }
    )
    return


@app.cell
def _(cp, mo):
    # CVX Options
    solver_ui = mo.ui.dropdown(label="Solver:",options=cp.installed_solvers(),value=cp.installed_solvers()[0])
    verbose_ui = mo.ui.checkbox(label="Verbose")
    optimize_ui = mo.vstack([solver_ui,verbose_ui])
    return optimize_ui, solver_ui, verbose_ui


@app.cell
def _(mo):
    # expansion costs
    substation_ui = mo.ui.slider(label="Substation ($/MVA)",value=50000,
                                 start=0,stop=100000,step=10000,
                                 show_value=True,debounce=True
                                )
    generation_ui = mo.ui.slider(label="Generation ($/MW):",value=500000,
                                 start=0,stop=2000000,step=100000,
                                 show_value=True,debounce=True)
    condenser_ui = mo.ui.slider(label="Condensers ($/MVAr):",value=50000,
                                start=0,stop=100000,step=10000,
                                show_value=True,debounce=True)
    capacitor_ui = mo.ui.slider(label="Capacitors ($/MW):",value=10000,
                                start=0,stop=100000,step=1000,
                                show_value=True,debounce=True)
    costs_ui = mo.vstack([substation_ui,generation_ui,condenser_ui,capacitor_ui])
    return capacitor_ui, condenser_ui, costs_ui, generation_ui, substation_ui


@app.cell
def _(angle_constraint_ui, mo, voltage_constraint_ui):
    # network options
    voltage_ui = mo.ui.slider(
        # start=0.001,
        # stop=0.1,
        # step=0.001,
        steps=[0.01,0.02,0.05,0.10,0.20,0.50,1.0],
        value=0.05,
        debounce=True,
        show_value=True,
        disabled=not voltage_constraint_ui.value,
    )
    angle_ui = mo.ui.slider(
        # start=1,
        # stop=30,
        # step=1,
        steps=[1,2,5,10,20,30,60,90,180],
        value=10,
        debounce=True,
        show_value=True,
        disabled=not angle_constraint_ui.value
    )
    return angle_ui, voltage_ui


@app.cell
def _(mo):
    voltage_constraint_ui = mo.ui.checkbox(value=True,label="Voltage limit (pu.V):")
    angle_constraint_ui = mo.ui.checkbox(value=True,label="Angle (+/-deg):")
    growth_constraint_ui = mo.ui.checkbox(value=False,label="Generation addition limit (pu):")
    return angle_constraint_ui, growth_constraint_ui, voltage_constraint_ui


@app.cell
def _(growth_constraint_ui, mo):
    growth_ui = mo.ui.slider(
        steps=[1.1,1.2,1.5,2.0,5.0,10.0],
        value=2.0,
        debounce=True,
        show_value=True,
        disabled=not growth_constraint_ui.value
    )
    return (growth_ui,)


@app.cell
def _(
    angle_constraint_ui,
    angle_ui,
    growth_constraint_ui,
    growth_ui,
    mo,
    voltage_constraint_ui,
    voltage_ui,
):
    network_ui = mo.vstack(
        [
            mo.hstack([voltage_constraint_ui,voltage_ui],justify='start'),
            mo.hstack([angle_constraint_ui,angle_ui],justify='start'),
            mo.hstack([growth_constraint_ui,growth_ui],justify='start'),
        ],
    )
    return (network_ui,)


@app.cell
def _(costs_ui, mo, network_ui, optimize_ui):
    mo.accordion({"## Problem Setup":mo.ui.tabs(
        {
            "**Costs**": costs_ui,
            "**Constraints**": network_ui,
            "**Optimizer**": optimize_ui,
        }
    )})
    return


@app.cell(hide_code=True)
def _(errors, mo, output, result, result_ui):
    mo.accordion({"## Solution": mo.ui.tabs({
        "**Results**":mo.vstack([result, result_ui,]),
        "**Output**" if output else "Output": mo.md(f"~~~\n{output}\n~~~" if output else "None"),
        "**Errors**" if errors else "Errors": mo.md(f"~~~\n{errors}\n~~~" if errors else "None"),
        }),
    })

    return


@app.cell
def _():
    return


@app.cell
def _(additions_ui, mo):
    mo.accordion(
        {
            "## Resource Additions": mo.vstack([additions_ui,])
        }
    )
    return


@app.cell
def _(mo, pd, postsolve, presolve):
    _gen = [
        sum(complex(round(x, 1), round(y, 1)) for x, y in z["gen"][:, 1:3])
        for z in (presolve, postsolve)
    ]
    _load = [
        sum(complex(round(x, 1), round(y, 1)) for x, y in z["bus"][:, 2:4])
        for z in (presolve, postsolve)
    ]
    _loss = [(x - y) for x, y in zip(_gen, _load)]
    _result = pd.DataFrame(
        {
            "Before OSP": [
                f"{_gen[0]:.1f}",
                f"{_loss[0]:.1f}",
                f"{_load[0]:.1f}",
            ],
            "After OSP": [
                f"{_gen[1]:.1f}",
                f"{_loss[1]:.1f}",
                f"{_load[1]:.1f}",
            ],
            "% Change": [
                f"{(1-_gen[1]/_gen[0])*-100:.1f}%",
                f"{(1-_loss[1]/_loss[0])*-100:.1f}%",
                f"{(1-_load[1]/_load[0])*-100:.1f}%",
            ],
        },
        index=["Generation dispatch (MVA)", "Line losses (MVA)", "Load (MVA)"],
    )
    mo.accordion(
        {
            "## Network Performance": mo.ui.table(
                _result,
                selection=None,
                show_data_types=False,
                text_justify_columns={
                    "Before OSP": "right",
                    "After OSP": "right",
                    "% Change": "right",
                },
            )
        }
    )
    return


@app.cell
def _(
    PPModel,
    PPSolver,
    angle_constraint_ui,
    angle_ui,
    array,
    capacitor_ui,
    condenser_ui,
    cp,
    file_ui,
    generation_ui,
    growth_constraint_ui,
    growth_ui,
    mo,
    np,
    solver_ui,
    sp,
    substation_ui,
    verbose_ui,
    voltage_constraint_ui,
    voltage_ui,
):
    mo.stop(file_ui.path(0) is None,"HINT: select a case file")
    model = PPModel(case=str(file_ui.path(0)))
    solver = PPSolver(model)
    solver.solve_opf()
    presolve_ok,presolve = solver.solve_pf(update="never",with_result=True)

    # model components
    bus = model.get_data("bus")
    branch = model.get_data("branch")
    gen = model.get_data("gen")
    gencost = model.get_data("gencost")
    graph = model._graph()

    # problem configuration
    load_margin = 0.2
    reference_bus = bus[bus.BUS_TYPE==3].BUS_I.values.tolist()
    voltage_limit = voltage_ui.value # pu.V
    substation_cost = substation_ui.value # $/MVA
    generation_cost = generation_ui.value # $/MW
    capacitor_cost = capacitor_ui.value # $/MVAr
    condenser_cost = condenser_ui.value # $/MVAr

    # model data
    def bus_i(x):
        if isinstance(x,list):
            return [bus_i(y) for y in x]
        return bus["BUS_I"].astype(int).tolist().index(int(x))
    N = len(bus)
    G = graph.laplacian(weighted=True,complex_flows=True).todense() # weighted graph Laplacian
    D = array([complex(*z) for z in bus[["PD","QD"]].values],ndmin=1)/model.case["baseMVA"] # demand
    I = graph.incidence(weighted=True,complex_flows=True).todense().T # weighted incidence matrix
    J = graph.incidence(weighted=False,complex_flows=False).todense().T # unweighted incidence matrix
    F = array(branch["RATE_A"].values,ndmin=1) # line ratings
    # problem variables
    if N > 0:
        x = cp.Variable(N,name='x') # bus voltage angles
        y = cp.Variable(N,name='y') # bus voltage magnitudes
        g = cp.Variable(N,name='g') # real power dispatch
        h = cp.Variable(N,name='h') # reactive power dispatch
        c = cp.Variable(N,name='c') # real power addition needed
        d = cp.Variable(N,name='d') # reactive power addition needed

        B = G.imag
        PD = D.real
        QD = D.imag

        _i,_j = [bus_i(x) for x in gen["GEN_BUS"]],array([0]*len(gen)) # generator bus index

        # generator limits
        PGmin = sp.sparse.coo_array((gen["PMIN"].values,(_i,_j)),shape=(N,1)).todense()/model.case["baseMVA"]
        PGmax = sp.sparse.coo_array((gen["PMAX"].values,(_i,_j)),shape=(N,1)).todense()/model.case["baseMVA"]
        QGmin = sp.sparse.coo_array((gen["QMIN"].values,(_i,_j)),shape=(N,1)).todense()/model.case["baseMVA"]
        QGmax = sp.sparse.coo_array((gen["QMAX"].values,(_i,_j)),shape=(N,1)).todense()/model.case["baseMVA"]

        # substation construction cost
        sub_cost = (substation_cost * cp.maximum(c-PGmax.T,0)) if substation_cost else 0.0

        # cost of real power capacity increases
        gen_cost = generation_cost * cp.abs(c) 

        # cost of reactive power capacity increases
        var_cost = (capacitor_cost-condenser_cost)/2*d + (condenser_cost+capacitor_cost)/2*cp.abs(d)

        # total cost of increases
        cost = cp.sum(sub_cost+gen_cost+var_cost)

        # constraints
        constraints = [
            B @ x == g + c - PD * ( 1 + load_margin ),  # KCL/KVL real power laws
            B @ y == h + d - QD * ( 1 + load_margin ),  # KCL/KVL reactive power laws

            x[bus_i(reference_bus)] == 0,  # swing bus voltage angle always 0
            y[bus_i(reference_bus)] == 1,  # swing bus voltage magnitude is always 1

            cp.abs(I @ x) <= F, # line flow limits

            # generation limits
            PGmin <= g, g <= PGmax, # real power limits
            QGmin <= h, h <= QGmax, # reactive power limits

            # real power capacity additions only positive and where existing generators are installed 
            c >= 0,
            ]
        if voltage_constraint_ui.value:
            constraints.append(cp.abs(y - 1) <= voltage_limit),  # limit voltage magnitude    
        if angle_constraint_ui.value:
            constraints.append(cp.abs(J @ x) <= np.pi/180*angle_ui.value), # +/-10 degree accuracy constraint
        if growth_constraint_ui.value:
            constraints.append(c <= g*growth_ui.value) # generation growth constraint

        with mo.capture_stderr() as _stderr:
            with mo.capture_stdout() as _stdout:
                problem = cp.Problem(cp.Minimize(cost), constraints)
                cp_config = dict(verbose=verbose_ui.value, 
                                 canon_backend=cp.SCIPY_CANON_BACKEND, 
                                 solver=solver_ui.value,
                                )
                try:
                    problem.solve(**cp_config)        
                    output = _stdout.getvalue()
                    errors = _stderr.getvalue()
                    solver.solve_opf()
                    postsolve_ok,postsolve = solver.solve_pf(with_result=True)
                except Exception as err:
                    output = "Solver failed"
                    errors = str(err)
                    postsolve = presolve
    else:
        problem = None
        output = "No data"
        errors = "No data"
        postsolve = presolve

    return (
        D,
        PGmax,
        c,
        d,
        errors,
        g,
        h,
        model,
        output,
        postsolve,
        presolve,
        problem,
        x,
        y,
    )


@app.cell
def _(busdata, mo, np, pd):
    additions_ui = mo.ui.table(
        pd.DataFrame(
            {
                "Total": [
                    busdata["Sub+"].sum(), 
                    busdata["Cap+"].sum(), 
                    busdata["Con+"].sum(), 
                    busdata["Gen+"].sum(),
                ],
                "Count": [
                    len(busdata[busdata["Sub+"]!=0]), 
                    len(busdata[busdata["Cap+"]!=0]), 
                    len(busdata[busdata["Con+"]!=0]),
                    len(busdata[busdata["Gen+"]!=0]),
                ],
                "Min": [
                    np.min(busdata.loc[busdata["Sub+"]!=0,"Sub+"]),
                    np.min(busdata.loc[busdata["Cap+"]!=0,"Cap+"]),
                    np.min(busdata.loc[busdata["Con+"]!=0,"Con+"]),
                    np.min(busdata.loc[busdata["Gen+"]!=0,"Gen+"]),
                ],
                "Max": [
                    np.max(busdata.loc[busdata["Sub+"]!=0,"Sub+"]), 
                    np.max(busdata.loc[busdata["Cap+"]!=0,"Cap+"]), 
                    np.max(busdata.loc[busdata["Con+"]!=0,"Con+"]),
                    np.max(busdata.loc[busdata["Gen+"]!=0,"Gen+"]),
                ],
            },
            index=[
                    "Substations",
                    "Capacitors", 
                    "Condensers", 
                    "Generators",
                ]
        ).round(1),
        # text_justify_columns={x:"right" for x in ["Total","Count","Min","Max"]},
        selection=None,
        show_data_types=False,
    )
    return (additions_ui,)


@app.cell
def _(errors, file_ui, pd, problem):
    if problem:
        result = pd.DataFrame({
          "Case": [file_ui.name(0)],
          "Status": [f"ERROR: {errors}" if problem.value is None else problem.status],
          "Cost": ["None" if problem.value is None else round(problem.value,3)],
          })
    else:
        result = pd.DataFrame({
          "Case": [file_ui.name(0)],
          "Status": ["No data"],
          "Cost": ["No data"],
          })
    return (result,)


@app.cell
def _(D, PGmax, c, cp, d, g, h, mo, model, np, pd, problem, x, y):
    if problem:
        if problem.value is None:
            result_ui = None
        elif not problem.status.startswith("optimal"):
            result_ui = mo.md(f"<font color=red>No solution data available when problem is {problem.status}</font>")
        else:
            busdata = pd.DataFrame({
                    "Bus":model.get_data("bus").BUS_I.values.astype(int),
                    "Va":(x.value*180/np.pi).round(2),
                    "Vm":y.value.round(3),
                    "Pg":g.value.round(3),
                    "Qg":h.value.round(3),
                    "Pd":D.real.round(3),
                    "Qd":D.imag.round(3),
                    "Sub+":cp.maximum(c-PGmax.T,0).value.round(3).tolist()[0],
                    "Gen+":c.value.round(3).tolist()[0],
                    "Con+":[max(-x,0) for x in d.value.round(3).tolist()],
                    "Cap+":[max(x,0) for x in d.value.round(3).tolist()],
                },
                )
            result_ui = mo.ui.table(busdata,
                        selection=None,
                        text_justify_columns={x:"right" for x in busdata.columns}
                       )
    else:
        busdata = mo.md("None")
        result_ui = mo.md("None")
    return busdata, result_ui


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import scipy as sp
    import cvxpy as cp
    import pandas as pd
    from numpy import array

    from ppmodel import PPModel
    from ppsolver import PPSolver

    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    pd.options.display.width = None
    return PPModel, PPSolver, array, cp, mo, np, pd, sp


if __name__ == "__main__":
    app.run()
