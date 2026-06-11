"""Optimal capacity expansion

This module implements a method for solving the problem of expanding real and
reactive power resources on an electric network as a relaxation of the
optimal power flow problem with softened constraints on loads (i.e., addition
of static VAR devices), generation capacities (i.e., generator and substation
capacity expansion), and line flows (i.e., transmission line capacity
expansion).

The relaxations enable the use of convex optimization solvers as illustrated
here using `cvxpy`. The softening of constraints converts the problem from a
canonical optimal powerflow problem to an optimal capacity expansion
problem.

The method does not support adding generators to PQ busses nor does it support
adding new transmission lines or transformers where none are already present
as one might find in an optimal sizing and placement problem.

Reference
---------

- https://github.com/eudoxys/fast_oce

"""
import os
from time import time
from copy import deepcopy as copy
from collections import namedtuple
from dataclasses import dataclass, field
from warnings import warn

import numpy as np
import scipy as sp
import cvxpy as cp

# from pypower.ppoption import ppoption
from pypower import idx_bus as bus
from pypower import idx_brch as branch
from pypower import idx_gen as gen
from pypower import idx_cost as cost
from pypower import idx_dcline
dcline = namedtuple("dcline",idx_dcline.c.keys())(**idx_dcline.c)
from ._violations import violations

@dataclass(kw_only=True,eq=False,order=False)
class OceOptions:

    cvx: dict = field(default_factory=dict)
    costs: dict[str,float] = field(default_factory=dict)
    margin: float = 0.15
    allin: bool = False
    setpoints: float|bool|None = None
    smallangles: float|list[float]|None = None
    """
    Properties
    ----------

    - `cvx` CVXpy solver options

    - `costs`: capacity addition costs (per-unit generation cost)
      Valid costs are
    
      - `"capacitor"`: cost of adding a capacitor (default is 0.1)
    
      - `"condensor"`: cost of adding a condensor (default is 1.0)
    
      - `"transformer"`: cost of increasing transformer capacity (default is 2.0)
    
      - `"powerline"`: cost of increasing powerline capacity (default is 10.0)

    - `margin`: load margin for sizing

    - `allin`: enable use of all available resources

    - `setpoints`: set voltage setpoints on generation busses (float or boolean)

    - `smallangles`: restrict voltage angles to near zero (float or array)
    """

    def __post_init__(self):
        def _setdefaults(item:dict,values:dict):
            for key,value in values.items():
                if key not in item:
                    item[key] = value
        _setdefaults(self.cvx,{
                "canon_backend": "SCIPY",
            })
        _setdefaults(self.costs,{
                "capacitor": 0.1, # $/MVAr
                "condensor": 1.0, # $/MVAr
                "transformer": 2.0, # $/MVA
                "powerline": 5.0, # $/MVA
            })

def runoce(
    data:dict,
    options:OceOptions|None=None,
    **kwargs
    ) -> dict:
    """Solve decoupled optimal capacity expansion problem
    
    Arguments
    ---------

    - `data`: `pypower` case data (see `pypower.casedata`)

    - `options`: OCE solver options class (see `pypower_sim.runoce.OceOptions`)

    - `**kwargs`: OCE solver options keyword (see `pypower_sim.runoce.OceOptions`)

    Returns
    -------

    - `dict`: solution results include the following:

      - `case`: a copy of the original problem data (see `pypower.casedata`)
      - `status`: status of the solution (see `cvxpy.Solve`)
      - `value`: value of the objection function (see `cvxpy.Solve`)
      - `objective`: objective function
      - `constraints`: constraints list
      - `problem`: cvxpy problem data (see `cvxpy.Problem`)
      - `solution`: solved case data (see `pypower.casedata`)
      - `parameters`: problem parameters (dict)
      - `variables`: problem variables (dict)
      - `ok`: valid solution obtained flag (boolean)
      - `updates`: list of updates to model

      In addition the following are included when the problem is feasible:
      
      - `pf`: real power flow on branches
      - `qf`: reactive power flow on branches
      - `vm`: bus voltage magnitudes
      - `va`: bus voltage angles
      - `pg`: real power generation dispatch
      - `qg`: reactive power generation dispatch
      - `ac`: capacitors/condensors additions
      - `ap`: real power generation capacity expansions
      - `aq`: reactive power generation capacity expansions
      - `al`: transformer and powerline capacity expansions
    """
    
    tic = time()

    # default options
    if options is None or kwargs != {}:
        options = OceOptions(kwargs) if kwargs else OceOptions()

    # model check
    assert "baseMVA" in data, "missing baseMVA value"
    assert "bus" in data, "missing bus array"
    assert "branch" in data, "missing branch array"
    assert "gen" in data, "missing gen array"

    # per-unit system
    puS = data["baseMVA"]

    # bus parameters
    N = len(data["bus"])
    assert N > 1, "too few busses"
    bb = data["bus"]
    vl = cp.Constant(value=bb[:, [bus.VMIN]], name="vl") # voltage lower limit
    vu = cp.Constant(value=bb[:, [bus.VMAX]], name="vu") # voltage upper limit
    pd = cp.Parameter(shape=(N,1), value=bb[:, [bus.PD]]/puS, name="pd") # load real power
    qd = cp.Parameter(shape=(N,1), value=bb[:, [bus.QD]]/puS, name="qd") # load reactive power
    bi = {i: n for n, i in enumerate(bb[:, bus.BUS_I])}  # bus index (i is not necessarily reasonable)

    # branch parameters
    M = len(data["branch"])
    assert M > 0, "too few branches"
    br = data["branch"] # branch data

    f_bus = [bi[x] for x in br[:,branch.F_BUS]]
    t_bus = [bi[x] for x in br[:,branch.T_BUS]]

    tap = br[:,[branch.TAP]].flatten()
    tap[np.where(tap==0)] = 1.0 # non-zero is only for transformers, zero is powerline (unity tap)
    err = np.where(tap<0)[0]
    assert len(err) == 0, f"bus[{err},TAP] < 0"

    shift = br[:,[branch.SHIFT]].flatten() * np.pi / 180

    br_status = br[:,[branch.BR_STATUS]].flatten()
    if options.allin:
        br_status[br_status==0] = 1
    err = np.where([x for x in br_status if x not in [0,1]])[0]
    assert len(err)==0, f"bus[{err},BR_STATUS] value is not in [0,1]"

    br_x = br[:,[branch.BR_X]].flatten()
    err = np.where(br_x==0)[0]
    assert len(err) == 0, f"bus[{err},BR_X] <= 0"

    x = br_status/br_x/tap

    b = sp.sparse.coo_matrix((x,(range(M),f_bus)),shape=(M,N)) \
        - sp.sparse.coo_matrix(((x+shift),(range(M),t_bus)),shape=(M,N)) 
    b = cp.Constant(value=b, name="b") # line susceptances

    f = sp.sparse.coo_matrix((br_status,(range(M),f_bus)),shape=(M,N)).T \
        - sp.sparse.coo_matrix((br_status,(range(M),t_bus)),shape=(M,N)).T 
    f = cp.Constant(value=f, name="f") # line connections

    s = br[:,[branch.RATE_A]]/puS
    s[s==0] = 1e5 # zero ratings are unlimited
    s = cp.Parameter(shape=(M,1),value=s,name="s") # line flow limits

    # gen parameters
    K = len(data["gen"])
    assert K > 0, "too few generators"
    gg = data["gen"]
    gi = np.array([bi[n] for n in gg[:,gen.GEN_BUS]])
    if options.allin:
        gg[:,gen.GEN_STATUS] = 1
    gs = gg[:,[gen.GEN_STATUS]]
    vg = cp.Parameter(shape=(K,1), value=gg[:,[gen.VG]], name="vg") # bus voltage setpoints
    pl = cp.Constant(value=gs*gg[:,[gen.PMIN]]/puS, name="pl") # real power minimum
    pu = cp.Constant(value=gs*gg[:,[gen.PMAX]]/puS, name="pu") # real power maximum
    ql = cp.Constant(value=gs*gg[:,[gen.QMIN]]/puS, name="ql") # reactive power minimum
    qu = cp.Constant(value=gs*gg[:,[gen.QMAX]]/puS, name="qu") # reactive power maximum
    g = sp.sparse.coo_matrix((np.ones(K),(list(range(K)),gi)),shape=(K,N)).T # sum generators to busses
    g = cp.Constant(value=g,name="g") # sum generators to busses

    # dc line parameters
    L = len(data["dcline"]) if "dcline" in data else 0
    if options.allin:
        data["dcline",dcline.BR_STATUS] = np.ones(K)
    if L > 0: # ignore DC line that are out of service
        dc_status = data["dcline"][:,dcline.BR_STATUS].flatten()
        dd = data["dcline"][dc_status!=0,:]
        L = dd.shape[0]
    if L > 0:
        f_bus = [bi[x] for x in dd[:,dcline.F_BUS]]
        t_bus = [bi[x] for x in dd[:,dcline.T_BUS]]
        da = cp.Constant(value=dd[:,[dcline.LOSS1]], name="da") # dcline loss function first-order term
        db = cp.Constant(value=dd[:,[dcline.LOSS0]], name="db")/puS # dcline loss function zero-order term
        df = sp.sparse.coo_matrix((np.ones(L),(range(L),f_bus)),shape=(L,N)) 
        df = cp.Constant(value=df.T, name="df") # from bus mapping
        dt = sp.sparse.coo_matrix((np.ones(L),(range(L),t_bus)),shape=(L,N))
        dt = cp.Constant(value=dt.T, name="dt") # to bus mapping
        dpmin = cp.Parameter(shape=(L,1),value=dd[:,[dcline.PMIN]],name="dpmin") / puS
        dpmax = cp.Parameter(shape=(L,1),value=dd[:,[dcline.PMAX]],name="dpmax") / puS
        dqminf = cp.Parameter(shape=(L,1),value=dd[:,[dcline.QMINF]],name="dqminf") / puS
        dqmaxf = cp.Parameter(shape=(L,1),value=dd[:,[dcline.QMAXF]],name="dqmaxf") / puS
        dqmint = cp.Parameter(shape=(L,1),value=dd[:,[dcline.QMINT]],name="dqmint") / puS
        dqmaxt = cp.Parameter(shape=(L,1),value=dd[:,[dcline.QMAXT]],name="dqmaxt") / puS

    # variables
    pf = cp.Variable((M,1), name="p")  # line real power flows
    qf = cp.Variable((M,1), name="q")  # line reactive power flows
    vm = cp.Variable((N,1), name="|v|", nonneg=True)  # voltage magnitudes
    va = cp.Variable((N,1), name="𝞱")  # voltage angles
    pg = cp.Variable((K,1), name="pg", nonneg=True)  # generator real power dispatch
    qg = cp.Variable((K,1), name="qg")  # generator reactive power dispatch
    if L > 0:
        dpf = cp.Variable(shape=(L,1), name="dpf",nonneg=True) # dc line real power from
        dpt = cp.Variable(shape=(L,1), name="dpt",nonneg=True) # dc line real power to
        dqf = cp.Variable(shape=(L,1), name="dqf") # dc line reactive power from
        dqt = cp.Variable(shape=(L,1), name="dqt") # dc line reactive power to

    # softened constraint variables
    ac = cp.Variable(shape=(N,1), name="ac") # capacitor/condensor additions
    ap = cp.Variable(shape=(K,1), name="ap", nonneg=True) # generator real power additions
    aq = cp.Variable(shape=(K,1), name="aq", nonneg=True) # generator reactive power additions
    al = cp.Variable(shape=(M,1), name="al", nonneg=True) # powerline/transformer capacity additions

    # sanity checks
    warnings = []

    # cannot handle pl>0 or dpmin > 0
    nz = np.where(pl.value>0)[0]
    warnings.extend([f"gen[{n}].PMIN > 0 is not supported, using zero" for n in nz])
    pl.value[nz,:] = 0
    if L > 0:
        nz = np.where(dpmin.value>0)[0]
        warnings.extend([f"dcline[{n}].PMIN > 0 is not supported, using zero" for n in nz])        
        dpmin.value[nz,:] = 0

    # bus vl/vu range
    if min(vu.value - vl.value) < 0 or min(vl.value) < 0.8 or max(vu.value) > 1.2:
        for n in np.where((vu.value - vl.value) < 0 )[0]:
            warnings.append(f"bus[{n},VMAX] < bus[{n},VMIN]")
        for n in np.where(vl.value < 0.8)[0]:
            warnings.append(f"bus[{n},VMIN] < 0.8")
        for n in np.where(vu.value > 1.2)[0]:
            warnings.append(f"bus[{n},VMAX] > 1.2")

    # line ratings
    if min(s.value) < 0:
        for n in np.where(s.value < 0)[0]:
            warnings.append(f"line[{n},RATE_A] < 0")

    # gen pl/pu range
    if min(pu.value - pl.value) < 0 or min(pu.value) < 0 or min(pl.value) < 0 :
        for n in np.where((pu.value - pl.value) < 0)[0]:
            warnings.append(f"gen[{n},PMAX] < gen[{n},PMIN]")
        for n in np.where(pl.value < 0)[0]:
            warnings.append(f"gen[{n},PMIN] < 0")
        for n in np.where(pu.value < 0)[0]:
            warnings.append(f"gen[{n},PMAX] < 0")

    # gen vg range
    if min(vg.value) < 0.8 or max(vg.value) > 1.2:
        for n in np.where(vg.value < 0.8)[0]:
            warnings.append(f"gen[{n}].vg < 0.8")
        for n in np.where(vg.value > 1.2)[0]:
            warnings.append(f"gen[{n}].vg > 1.2")

    # warn of check failures
    if warnings:
        warn(f"{len(warnings)} model warnings (see 'warnings' for details)")

    # setup Feasible Sets
    ref = [n for n, x in enumerate(data["bus"][:, bus.BUS_TYPE]) if x == 3] # reference bus(ses)
    nongen = list(set(range(N)) - set(gi)) # non-generation busses
    powerlines = [n for n,x in enumerate(data["branch"][:,branch.TAP]) if x == 0]
    transformers = list(set(range(M))- set(powerlines))

    # cost function
    cost = cp.sum(ap) # + cp.sum(aq)/10 # generation capacity costs
    cost += cp.sum( # capacity/condensor costs
            ( options.costs["capacitor"] - options.costs["condensor"] ) * ac / 2
            + ( options.costs["capacitor"] + options.costs["condensor"] ) * cp.abs(ac) / 2
            )
    cost += options.costs["powerline"] * cp.sum(al[powerlines]) # powerline costs
    cost += options.costs["transformer"] * cp.sum(al[transformers]) # transformer costs

    # constraints
    constraints = [

        # Feasible Set 2
        pf == b @ va, # Equation (1a)
        qf == b @ vm, # Equation (1b)

        # Feasible Set 4
        pl <= pg, pg <= pu + ap, # Equation (3c)
        ql - aq <= qg, qg <= qu + aq, # Equation (3d)
        cp.abs(pf) <= s + al, # Equation (4b)
        cp.abs(qf) <= s + al, 
        cp.abs(pf) + cp.abs(qf) <= 1.4 * ( s + al ),
        vl <= vm, vm <= vu, # Equation (5b)

        # practical constraints not specified in the mathematical model
        va[ref] == 0,  # reference bus angle is always 0

        # constraints on addition placements
        ac[gi] == 0, # no capacitors/condensors at generation busses
        
        # limits on reactive power additions relative to real power additions
        cp.abs(qu) + aq <= pu + ap,
        cp.abs(ql) + aq <= pu + ap,
    ]

    # dc lines
    if L == 0:
        constraints += [ # line flows without DC lines
            f @ pf + pd*(1+options.margin) == g @ pg, # Equation (2a)
            f @ qf + qd*(1+options.margin) + ac == g @ qg, # Equation (2b)
            ]
    else:
        constraints += [ # line flows with DC lines
            f @ pf + pd*(1+options.margin) + df @ dpf == g @ pg + dt @ dpt, # Equation (2a)
            f @ qf + qd*(1+options.margin) + ac + df @ dqf == g @ qg + dt @ dqt, # Equation (2b)
            dpt == cp.multiply(1-da,dpf) - db, # DC losses
            # dpmin <= dpf, # non-zero not supported
            0 <= dpf, dpf <= dpmax, # real power DC "to" injection limits
            dqminf <= dqf, dqf <= dqmaxf, # reacation power DC "from" injection limits
            dqmint <= dqt, dqt <= dqmaxf, # real power DC "to" injection limits
            ]

    # small angle assumption
    if not options.smallangles is None:
        constraints.append(cp.abs(va) <= options.smallangles)  # +/- 10 degrees for decoupling assumptions to be valid

    # bus voltage setpoints
    if isinstance(options.setpoints,float):
        constraints.append(vm[ref] == options.setpoints)
    elif options.setpoints is True:
        constraints.append(vm[gi] == vg)

    # problem statement
    objective = cp.Minimize(cost)
    problem = cp.Problem(objective,constraints)
    problem.solve(**options.cvx)

    # solution results
    result = {
        "ok": False,
        "case": copy(data),
        "status": problem.status,
        "value": np.round(problem.value,4),
        "problem": problem,
        "objective": objective,
        "constraints": constraints,
        "constants": {
            "b (pm.S)": b.value.todense(), # TODO: remove todense
            "f (pu)": f.value.todense(), # TODO: remove todense
            "g (pu)": g.value.todense(), # TODO: remove todense
            "vl (pu.kV)": vl.value.T[0],
            "vu (pu.kV)": vu.value.T[0],
            "pl (pu.MW)": pl.value.T[0],
            "pu (pu.MW)": pu.value.T[0],
            "ql (pu.MVAr)": ql.value.T[0],
            "qu (pu.MVAr)": qu.value.T[0],
        },
        "parameters": {
            "s (pu.MVA)": s.value.T[0],
            "pd (pu.MW)": pd.value.T[0],
            "qd (pu.MVAr)": qd.value.T[0],
            "vg (pu.kV)": vg.value.T[0],
        },
        "solution": {},
        "violations": {},
        "warnings": warnings
    }
    if problem.status == "optimal":

        result["variables"] = {
            "pf (pu.MW)": pf.value.T[0],
            "qf (pu.MVAr)": qf.value.T[0],
            "vm (pu.kV)": vm.value.T[0],
            "va (deg)": (va.value*180/np.pi).T[0],
            "pg (pu.MW)": pg.value.T[0],
            "qg (pu.MVAr)": qg.value.T[0],
            "ac (pu.MVAr)": ac.value.T[0],
            "ap (pu.MVAr)": ap.value.T[0],
            "aq (pu.MVAr)": aq.value.T[0],
            "al (pu.MVAr)": al.value.T[0],
        }

        # creates solution
        solution = copy(data)
        
        # bus updates
        solution["bus"][:,bus.VA] = va.value.T[0] * 180 / np.pi
        solution["bus"][:,bus.VM] = vm.value.T[0]
        solution["bus"][:,bus.BS] = solution["bus"][:,bus.BS] + ac.value.T[0]

        # branch updates
        if options.allin:
            solution["branch"][:,branch.BR_STATUS] = np.ones(M)
        if solution["branch"].shape[1] <= branch.PF:
            solution["branch"] = np.append(solution["branch"],pf.value*puS,axis=1)
        else:
            solution["branch"][:,[branch.PF]] = pf.value * puS
        if solution["branch"].shape[1] <= branch.QF:
            solution["branch"] = np.append(solution["branch"],qf.value*puS,axis=1)
        else:
            solution["branch"][:,[branch.QF]] = qf.value * puS
        rows = np.where((solution["branch"][:,[branch.RATE_A]]>0) & (al.value>0) )[0]
        ratio = 2 * al.value.flatten()[rows] * puS / solution["branch"][rows,branch.RATE_A].flatten() + 1
        for column in [branch.RATE_A,branch.RATE_B,branch.RATE_C]: # raised values
            solution["branch"][rows,column] = solution["branch"][rows,column] * ratio
        for column in [branch.BR_R,branch.BR_X,branch.BR_B]: # lowered values
            solution["branch"][rows,column] = solution["branch"][rows,column] / ratio
        
        # generator updates
        if options.allin:
            solution["gen"][:,gen.GEN_STATUS] = np.ones(K)
        solution["gen"][:,[gen.PG]] = pg.value * puS
        solution["gen"][:,[gen.QG]] = qg.value * puS
        solution["gen"][:,[gen.PMAX]] = solution["gen"][:,[gen.PMAX]] + ap.value * puS
        solution["gen"][:,[gen.QMIN]] = solution["gen"][:,[gen.QMIN]] - aq.value * puS
        solution["gen"][:,[gen.QMAX]] = solution["gen"][:,[gen.QMAX]] + aq.value * puS

        result["solution"] = solution

        # create update list
        updates = []
        for n,x in enumerate([x for x in ap.value[:,0]*puS]):
             if abs(x) > 1e-3:
                updates.append(f"add {x:.3f} MW gen[{n},PMAX]")
        for n,x in enumerate([x for x in aq.value[:,0]*puS]):
             if abs(x) > 1e-3:
                updates.append(f"add {x:.3f} MVAr gen[{n},QMAX]")
        for n,x in enumerate([x for x in ac.value[:,0]*puS]):
             if abs(x) > 1e-3:
                updates.append(f"add {x:.3f} MVAr to bus[{n},BS]")
        for n,x in enumerate([x for x in al.value[:,0]*puS]):
             if abs(x) > 1e-3:
                updates.append(f"add {x:.3f} MVA to branch[{n},RATE_A]")
        result["updates"] = updates

        # update dcline
        if L > 0: 
            dd = solution["dcline"]
            di = np.where(dd[:,dcline.BR_STATUS] == 1)[0]
            dd[di,dcline.PF] = dpf.value.T[0] * puS
            dd[di,dcline.QF] = dqf.value.T[0] * puS
            dd[di,dcline.PT] = dpt.value.T[0] * puS
            dd[di,dcline.QT] = dqt.value.T[0] * puS
            # dd[:,dcline.PMIN] = np.zeros(shape=len(dd))
            # dd[:,[dcline.VF]] = dvf.value
            # dd[:,[dcline.VT]] = dvt.value

        # create violations list
        checks = violations(solution)
        if checks:
            result["violations"] = checks
        
        result["ok"] = True

    toc = time()
    result["time"] = round(toc-tic,3)
    return result

if __name__ == "__main__":

    from ppmodel import PPModel
    from ppsolver import PPSolver
    from _violations import violations

    path = "../test"
    n_error = 0
    n_warning = 0
    for file in sorted([x for x in os.listdir(path) if x.startswith("case") and x.endswith(".py")]):
        name = file[:-3]
        model = PPModel(name,case=f"{path}/{file}")
        solvers = PPSolver(model)

        print(f"Testing {name}",end="...")
        if not solvers.solve_opf(use_acopf=False):
            oce = runoce(model.case)
            if len(oce["warnings"]):
                n_warning += len(oce["warnings"])
                print(f"{len(oce["warnings"])} warnings")
                for warning in oce["warnings"]:
                    print(f"WARNING [{file}]: OCE {warning}")
            model.case = oce["solution"]
            ok,pf = solvers.solve_pf(with_result=True)
            errs = violations(pf)
            if not ok:
                n_error += 1
                print(f"1 error\nERROR [{file}]: PF failed")
            elif errors := sum([len(x) for x in errs.values()]):
                n_warning += errors
                print(f"WARNING [{file}]: PF {errors} violations")
            elif "warnings" in pf and len(pf["warnings"]) > 0:
                n_warning += len(pf["warnings"])
                print(f"{len(pf["warnings"])} warnings")
                for warning in pf["warnings"]:
                    print(f"WARNING [{file}]: PF {warning}")
            else:
                print("ok")

        else:
            print("OPF ok")

    print(f"Tests complete: {n_error} errors and {n_warning} warnings")
