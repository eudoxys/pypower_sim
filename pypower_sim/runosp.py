r"""Optimal resource sizing/placement

Optimal sizing and placement (OSP) finds the lowest cost set of additional
substations, generators, condensers, capacitors, and powerline ratings needed
to make a network model feasible and operate with a minimum of losses given a
maximum load.

For an $N$-bus and $M$-line network the problem is stated as 

$\underset{x,y,g,h,c,d}{\min} (c-\Re \hat{G}) \lfloor_0 ~ C_s + c~G_g + \frac12(C_c-C_d)~d + \frac12(C_c+C_d)~|d| + C_l ~ f$

subject to

- $g+c-\Re D~(1+L)-B~x = 0$ (KCL/KVL real power balance)

- $h+d-\Im D~(1+L)-B~y = 0$ (KCL/KVL reactive power balance)

- $x_{ref} = 0$ (swing bus angle set to 0)

- $y_{ref} = 1$ (swing bus magnitude set to 1)

- $\Re \check G \le g \le \Re \hat G$ (real power generation limits)

- $\Im \check G \le h \le \Im \hat G$ (reactive power generation limits)

- $c \ge 0$ (generation real power can only be positive)

- $|y-1| \le V_m$ (voltage magnitude limits)

- $|J~x| \le V_a$ (voltage angle limits)

- $c \le C$ (generation capacity expansion limits)

- $I~x \le F + f$ (line flow limits)

- $f \ge 0$ (powerline rating upgrade can only be positive)

- $f \le K$ (powerline rating upgrade limits, if any)

where the variable

- $c \in \mathbb {R}^N$ is the new real power generation added to provide
  sufficient real power to meet the load plus the load margin,

- $d \in \mathbb {R}^N$ is the new reactive power support devices added to
  provide sufficient reactive power to minimize line losses and meet the load
  plus the load margin,

- $f \in \mathbb {R}^M$ is the adding line rating needed to make the powerflow
  feasible given the load, including the load margin,

- $g \in \mathbb {R}^N$ is the real power generation dispatched to meet the
  load plus the load margin,

- $h \in \mathbb {R}^N$ is the reactive power generation dispatched to meet
  the load plus the load margin,

- $x \in \mathbb {R}^N$ is the bus voltage angle in radians,

- $y \in \mathbb {R}^N$ is the bus voltage magnitude per-unit kV,

the parameter

- $B \in \mathbb {R}^{N \times N}$ is the weighted network graph Laplacian
  based on the line susceptance per-unit Siemens,

- $C \in \mathbb {R}^N$ is the generation addition capacity limits in MW, where
  $C_n=0$ when bus $n$ is a PQ bus.

- $D \in \mathbb {C}^N$ is the complex load per-unit base MVA,

- $F \in \mathbb {R}^M$ is the line flow limits per-unit base MVA,

- $\hat G \in \mathbb {C}^N$ is the maximum complex power generation per-unit base MVA,

- $\check G \in \mathbb {C}^N$ is the minimum complex power generation per-unit base MVA,

- $I \in \mathbb {R}^{M \times N}$ is the weighted network incidence matrix
  based on the line susceptance per-unit Siemens,

- $J \in \mathbb {R}^{M \times N}$ is the unweighted network incidence matrix,

- $K \in \mathbb {R}^M$ is the line rating upgrade limits,

- $V_a \in \mathbb{R}$ is the voltage angle limit in radians,

- $V_m \in \mathbb{R}$ is the voltage magnitude limit per unit kV,

and the cost

- $C_c \in \mathbb{R}$ is the cost of adding capacitors in $/MVAr,

- $C_d \in \mathbb{R}$ is the cost of adding condensers in $/MVAr,

- $C_g \in \mathbb{R}$ is the cost of adding generators in $/MW,

- $C_l \in \mathbb{R}$ is the cost of upgrading a line rating in $/MW,

- $C_s \in \mathbb{R}$ is the cost of adding substations in $/MVA,

# Caveats

The OSP algorithm cannot increase line susceptance in cases where the OPF
would become infeasible given the specified load margin and resources
available, even when the OPF is feasible under the same conditions. As a
result, the OSP problem may be infeasible due to excessive reactive power
losses on the lines. The only solution available at this time is to increase
powerline susceptance on lines where the losses are too high and adjust line
admittance accordingly.

# References

- [Joshua Taylor, *Convex Optimization of Power Systems*, Cambridge University
  Press (2015)](https://books.google.com/books?hl=en&lr=&id=JBdoBgAAQBAJ&oi=fnd&pg=PR11&dq=info:4_zKJR2GVGAJ:scholar.google.com&ots=A23AB6jlr9&sig=D2uoDpJMlNfCT9an9WOMuBvfk_k#v=onepage&q&f=false)

- [CVXPY](https://www.cvxpy.org/index.html)
"""

from typing import TypeVar
from copy import deepcopy

from numpy import array
import numpy as np
import scipy as sp
import cvxpy as cp

class OspConfig:
    """OSP solver configuration parameters"""
    def __init__(self,**kwargs):
        """Construct an OSP solver configuration"""

        self.substation_cost : float=250e3
        """Substation installation cost (defaults to 250,000 $/MVA)"""

        self.generation_cost : float=7500e3
        """Generation capacity expansion cost (defaults to 750,000 $/MW)"""

        self.condenser_cost : float=400e3
        """Condenser capacity expansion cost (defaults to 400,000 $/MVAr)"""

        self.capacitor_cost : float=25e3
        """Capacitor capacity expansion costs (defaults to 25,000 $/MW)"""

        self.powerline_cost : float = 400e3 # roughly $2M/mile for 500 MW over 100 miles
        """Powerline capacity expansion costs (defaults to 400,000 $/MW)"""

        self.load_margin : float = 0.2
        """Load capacity margin (defaults to 0.2 pu.MW)"""

        self.reference_bus : int|list[int] = None
        """Reference bus id (defaults to `None`, i.e., `BUS_TYPE=3`)"""

        self.reference_voltage : complex = complex(1,0)
        """Reference bus voltage (defaults to $1+0j$ pu.kV)"""

        self.voltage_limit : float = 0.05
        """Voltage magnitude deviation limit (in pu.kV or `None`, defaults to 0.05 pu.kV)"""

        self.angle_limit : float = 10.0
        """Voltage angle deviation limit (in deg or `None`, defaults to 10 deg)"""

        self.generation_limit : float|list[float] = None
        """Generation addition limit (in MW, defaults to `None`)"""

        self.line_rating : str = "RATE_A"
        """Specify line rating field to use"""

        self.powerline_limit : float|list[float] = None
        """Powerline rating upgrade limit (in MW, defaults to `None`)"""

        self.cvx_solver = {
            "solver": "CLARABEL",
            "verbose": False,
            "canon_backend": "SCIPY",
        }
        """CVX solver configuration"""

        self.results = {
            "value": "value",
            "status": "status",
            "problem": "problem",
            "generators": "generators",
            "capacitors": "capacitors",
            "condensers": "condensers",
            "voltages": "voltages",
            "lines": "lines",
            "error": "error",
        }
        """Results to return from call to `pypower_sim.runosp.runosp`"""

        for key,value in kwargs.items():
            assert hasattr(self,key), f"{repr(key)} is not a valid OspConfig attribute name"
            values = getattr(self,key)
            if isinstance(values,dict):
                assert isinstance(value,dict), f"{key}={repr(value)} value must be a dict"
                for subkey,subvalue in value.items():
                    assert subkey in values, f"{subkey}={repr(subvalue)} key not in {key} dict"
                    values[subkey] = subvalue
                    setattr(self,key,values)
            else:
                setattr(self,key,value)

    def __repr__(self):
        data = [f"{x}={repr(getattr(self,x))}" for x in dir(self) if not x.startswith("_")]
        return f"OspConfig({','.join(data)})"

def runosp(
    model:TypeVar('pypower_sim.ppmodel.PPModel'),
    config:OspConfig|dict=None,
    ) -> dict:
    """Solve the optimal sizing/placement problem

    # Arguments

    - `model`: a `pypower_sim.ppmodel.PPModel` object

    - `config`: OSP configuration options

    # Returns

    - `dict`: See `pypower_sim.runosp.OspConfig.results` for return data configuration
    """

    if config is None:
        config = OspConfig()
    elif isinstance(config,dict):
        config = OspConfig(**config)
    else:
        assert isinstance(config,OspConfig), f"{config=} is not an OspConfig object"

    # model components
    bus = model.get_data("bus")
    branch = model.get_data("branch")
    gen = model.get_data("gen")
    graph = model._graph()

    N = len(bus)
    M = len(branch)

    assert N > 0, "no busses"
    assert M > 0, "no branches"

    def bus_i(x):
        """Get bus index from bus id"""
        if isinstance(x,list):
            return [bus_i(y) for y in x]
        return bus["BUS_I"].astype(int).tolist().index(int(x))
        
    G = graph.laplacian(weighted=True,complex_flows=True).todense() # weighted graph Laplacian
    D = array([complex(*z) for z in bus[["PD","QD"]].values],ndmin=1) / model.case["baseMVA"] # demand
    I = graph.incidence(weighted=True,complex_flows=True).todense().T # weighted incidence matrix
    J = graph.incidence(weighted=False,complex_flows=False).todense().T # unweighted incidence matrix
    F = array(branch[config.line_rating].values,ndmin=1) / model.case["baseMVA"] # line ratings

    # reference bus and reference voltage
    reference_bus = graph.refbus if config.reference_bus is None else config.reference_bus

    if config.cvx_solver["verbose"]:
        print(f"{G=}")
        print(f"{D=}")
        print(f"{I=}")
        print(f"{J=}")
        print(f"{F=}")

    x = cp.Variable(N,name='x') # bus voltage angles
    y = cp.Variable(N,name='y') # bus voltage magnitudes
    g = cp.Variable(N,name='g') # real power dispatch
    h = cp.Variable(N,name='h') # reactive power dispatch
    c = cp.Variable(N,name='c') # real power addition needed
    d = cp.Variable(N,name='d') # reactive power addition needed
    f = cp.Variable(M,name='f') # line flow capacity expansion

    B = G.imag
    PD = D.real
    QD = D.imag

    # generator limits
    i,j = [bus_i(x) for x in gen["GEN_BUS"]],array([0]*len(gen)) # generator bus index
    PGmin = sp.sparse.coo_array((gen["PMIN"].values,(i,j)),shape=(N,1)).todense()/model.case["baseMVA"]
    PGmax = sp.sparse.coo_array((gen["PMAX"].values,(i,j)),shape=(N,1)).todense()/model.case["baseMVA"]
    QGmin = sp.sparse.coo_array((gen["QMIN"].values,(i,j)),shape=(N,1)).todense()/model.case["baseMVA"]
    QGmax = sp.sparse.coo_array((gen["QMAX"].values,(i,j)),shape=(N,1)).todense()/model.case["baseMVA"]

    # substation construction cost
    sub_cost = config.substation_cost * cp.maximum(c-PGmax.T,0)

    # cost of real power capacity additions
    gen_cost = config.generation_cost * cp.abs(c) 

    # cost of reactive power capacity additions
    svd_cost = (config.capacitor_cost-config.condenser_cost)/2*d + (config.condenser_cost+config.capacitor_cost)/2*cp.abs(d)

    # total cost of capacity additions
    cost = cp.sum(sub_cost + gen_cost + svd_cost)

    # cost of line rated capacity expansion
    if config.powerline_limit is not None:

        pwl_cost = config.powerline_cost * f
        cost += sum(pwl_cost)

    # constraints
    constraints = [
        B @ x == g + c - PD * ( 1 + config.load_margin ),  # KCL/KVL real power laws
        B @ y == h + d - QD * ( 1 + config.load_margin ),  # KCL/KVL reactive power laws

        x[reference_bus] == np.angle(config.reference_voltage),  # swing bus(ses) voltage angle value(s)
        y[reference_bus] == np.abs(config.reference_voltage),  # swing bus(ses) voltage magnitude value(s)

        # generation limits
        PGmin <= g, g <= PGmax, # real power limits
        QGmin <= h, h <= QGmax, # reactive power limits

        c >= 0, # only real power generation additions allowed
        ]

    # voltage magnitudes are constrained
    if config.voltage_limit is not None:

        constraints.append(cp.abs(y - 1) <= config.voltage_limit),

    # voltage angle are constrainted (recommend no more than +/-10 deg)
    if config.angle_limit is not None:

        constraints.append(cp.abs(J @ x) <= np.pi/180*config.angle_limit),

    # generation growth constraints specified (bus types may need to be changed by caller)
    if config.generation_limit is not None:

        constraints.append(c <= config.generation_limit)

    # only allow generation to be added at non-PQ busses
    else: 

        k = bus[bus.BUS_TYPE == 1].index.astype(int).tolist()  # PQ bus list
        constraints.append(c[k] == 0)

    # line rating upgrades allowed
    if config.powerline_limit is not None:

        cp.abs(I @ x) <= F + f, # line flow limits
        f >= 0, 

    # no line rating upgrades permitted
    else:

        cp.abs(I @ x) <= F, # line flow limits

    problem = cp.Problem(cp.Minimize(cost), constraints)

    # construct result
    error = None
    value = None
    status = False
    generators = None
    capacitors = None
    condensers = None
    voltages = None
    lines = None
    try:

        problem.solve(**config.cvx_solver)        
        value = problem.value
        status = problem.status.startswith("optimal")
        generators = c.value
        if d.value is None:
            error = problem.status
        else:
            capacitors = np.clip(d.value,a_min=0,a_max=None)
            condensers = np.clip(-d.value,a_min=0,a_max=None)
            voltages = y.value * np.exp(x.value*1j)
            lines = f.value

    except Exception as err:

        error = str(err)

    result = {x:eval(y) for x,y in config.results.items()}

    # print(result["lines"])
    return result

if __name__ == "__main__":

    import os, sys
    import time
    import warnings

    from ppmodel import PPModel
    from ppsolver import PPSolver

    import pandas as pd

    pd.options.display.max_columns = None
    pd.options.display.width = None
    pd.options.display.max_rows = None

    import numpy as np
    np.set_printoptions(formatter={
        "float_kind":lambda x:f"{x:8.3f}",
        "complex_kind":lambda x:f"{x.real:8.3f}{x.imag:+8.3f}j",
        })

    error_dump = False
    error_stop = False

    def intx(x,with_tail=False):
        """Return int value with tail string"""
        import re
        match = re.search(r"\d+",x)
        if match:
            result = match.group()
            if with_tail:
                return int(result),x[len(result):]
            return int(result)
        raise ValueError("not an integer")
    tests = sorted([x[4:-3] for x in os.listdir("../test") if x.startswith("case") and x.endswith(".py")],key=lambda x:intx(x,True))
    sz = max([len(x) for x in tests])
    testcalls = {
                "Initial PF": "solve_pf",
                "Initial OPF": "solve_opf",
                "Optimal PF": "solve_pf",
                "Optimal Sizing":"solve_osp",
                "Sized OPF":"solve_opf",
                "Sized PF":"solve_pf",
                }
    print(f"{'Case':^20s}",*[f" {x} " for x in testcalls],f"{'Time (s)':^8s}",f"{'Newgen (%MW)':^14s}",f"{'Savings (%MW)':^14s}")
    print("-"*20,*["-"*(len(x)+2) for x in testcalls],"--------","--------------","--------------")
    reportlist = []
    for caseid in tests:
        case = f"../test/case{caseid}.py"
        print(f"case{caseid}.py"," "*(20-len(caseid)-7),end="",flush=True)

        try:
            model = PPModel(os.path.splitext(os.path.basename(case))[0],case=case)
            assert "gencost" in model.case, "no gencost data"
        except Exception as err:
            reportlist.append((case,"model","exception",str(err)))
            print(f"*** unusable model: {err} ***",flush=True)
            continue
        tic = time.time()
        solver = PPSolver(model)
        pwrtotal = []
        gentotal = []
        for label,call in testcalls.items():
            try:
                with warnings.catch_warnings(record=True) as recording:
                    warnings.simplefilter("always")
                    status,result = getattr(solver,call)(with_result=True)
                report = "ok"
                if recording:
                    report = "warning"
                    for msg in recording:
                        reportlist.append((case,label,"warning",f"{msg.category.__name__} -- {msg.message}" ))
                if status == False:
                    if "error" in result:
                        report = "error"
                        reportlist.append((case,label,"error",result["error"]))
                    else:
                        report = "failed"
                        reportlist.append((case,label,"failed",result["raw"]["output"]["message"]))
                elif call == "solve_pf":
                    pwrtotal.append(sum((abs(complex(x,y)) for x,y in model.get_data("gen")[["PMAX","QMAX"]].values)))
                    gentotal.append(sum((abs(complex(x,y)) for x,y in model.get_data("gen")[["PG","QG"]].values)))
            except Exception as err:
                report = "exception"
                reportlist.append((case,label,"exception",err))
                model.print()
                raise
            print(" "*((len(label)-len(report))//2),report," "*((len(label)+1-len(report))//2),end=" ",flush=True)
        toc = time.time()
        print(f"{toc-tic:8.3f}",f"{(1-pwrtotal[1]/pwrtotal[2])*100:12.2f}% ",f"{(1-gentotal[2]/gentotal[1])*100:12.2f}% ")
    print("-"*20,*["-"*(len(x)+2) for x in testcalls],"--------","--------------","--------------")

    print("","Output details","==============",sep="\n")
    for case,label,event,report in reportlist:
        print(f"{event.upper()} [{case}@{label}]: {report}")
