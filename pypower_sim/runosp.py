r"""Optimal resource sizing/placement

Optimal sizing and placement (OSP) finds the lowest cost set of additional
substations, generators, condensers, and capacitors needed to make a network
model feasible and operate with a minimum of losses given a maximum load.

For an $N$-bus and $M$-line network the problem is stated as 

$\underset{x,y,g,h,c,d}{\min} (c-\Re \hat{G}) \lfloor_0 ~ C_s + c~G_g + \frac12(C_c-C_d)~d + \frac12(C_c+C_d)~|d|$

subject to

- $g+c-\Re D~(1+L)-B~x = 0$ (KCL/KVL real power balance)

- $h+d-\Im D~(1+L)-B~y = 0$ (KCL/KVL reactive power balance)

- $x_{ref} = 0$ (swing bus angle set to 0)

- $y_{ref} = 1$ (swing bus magnitude set to 1)

- $I~x \le F$ (line flow limits)

- $\Re \check G \le g \le \Re \hat G$ (real power generation limits)

- $\Im \check G \le h \le \Im \hat G$ (reactive power generation limits)

- $c \ge 0$ (generation real power can only be positive)

- $|y-1| \le V_m$ (voltage magnitude limits)

- $|J~x| \le V_a$ (voltage angle limits)

- $c \le C$ (generation capacity expansion limits)

where the variable

- $c \in \mathbb {R}^N$ is the new real power generation added to provide
  sufficient real power to meet the load plus the load margin,

- $d \in \mathbb {R}^N$ is the new reactive power support devices added to
  provide sufficient reactive power to minimize line losses and meet the load
  plus the load margin,

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

- $V_a \in \mathbb{R}$ is the voltage angle limit in radians,

- $V_m \in \mathbb{R}$ is the voltage magnitude limit per unit kV,

and the cost

- $C_c \in \mathbb{R}$ is the cost of adding capacitors in $/MVAr,

- $C_d \in \mathbb{R}$ is the cost of adding condensers in $/MVAr,

- $C_g \in \mathbb{R}$ is the cost of adding generators in $/MW,

- $C_s \in \mathbb{R}$ is the cost of adding substations in $/MVA,

# References

- [Joshua Taylor, *Convex Optimization of Power Systems*, Cambridge University
  Press (2015)]
  (https://books.google.com/books?hl=en&lr=&id=JBdoBgAAQBAJ&oi=fnd&pg=PR11&dq=info:4_zKJR2GVGAJ:scholar.google.com&ots=A23AB6jlr9&sig=D2uoDpJMlNfCT9an9WOMuBvfk_k#v=onepage&q&f=false)

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

        self.substation_cost:float=250000
        """Substation installation cost (defaults to 250,000 $/MVA)"""

        self.generation_cost:float=750000
        """Generation capacity expansion cost (defaults to 750,000 $/MW)"""

        self.condenser_cost:float=400000
        """Condenser capacity expansion cost (defaults to 400,000 $/MVAr)"""

        self.capacitor_cost:float=25000
        """Capacitor capacity expansion costs (defaults to 25,000 $/MW)"""

        self.load_margin: float = 0.2
        """Load capacity margin (defaults to 0.2 pu.MW)"""

        self.reference_bus: int|list[int] = None
        """Reference bus id (defaults to `None`, i.e., `BUS_TYPE=3`)"""

        self.reference_voltage: complex = complex(1,0)
        """Reference bus voltage (defaults to $1+0j$ pu.kV)"""

        self.voltage_limit: float = 0.05
        """Voltage magnitude deviation limit (in pu.kV or `None`, defaults to 0.05 pu.kV)"""

        self.angle_limit: float = 10.0
        """Voltage angle deviation limit (in deg or `None`, defaults to 10 deg)"""

        self.generation_limit: float|list[float] = None
        """Generation addition limit (in MW, defaults to `None`)"""

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
    D = array([complex(*z) for z in bus[["PD","QD"]].values],ndmin=1)/model.case["baseMVA"] # demand
    I = graph.incidence(weighted=True,complex_flows=True).todense().T # weighted incidence matrix
    J = graph.incidence(weighted=False,complex_flows=False).todense().T # unweighted incidence matrix
    F = array(branch["RATE_A"].values,ndmin=1) / model.case["baseMVA"] # line ratings

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

    # constraints
    constraints = [
        B @ x == g + c - PD * ( 1 + config.load_margin ),  # KCL/KVL real power laws
        B @ y == h + d - QD * ( 1 + config.load_margin ),  # KCL/KVL reactive power laws

        x[reference_bus] == np.angle(config.reference_voltage),  # swing bus(ses) voltage angle value(s)
        y[reference_bus] == np.abs(config.reference_voltage),  # swing bus(ses) voltage magnitude value(s)

        cp.abs(I @ x) <= F, # line flow limits

        # generation limits
        PGmin <= g, g <= PGmax, # real power limits
        QGmin <= h, h <= QGmax, # reactive power limits

        # real power capacity additions only positive and where existing generators are installed 
        c >= 0,
        ]

    if config.voltage_limit: # limit voltage magnitude    

        constraints.append(cp.abs(y - 1) <= config.voltage_limit),

    if config.angle_limit: # +/-10 degree accuracy constraint

        constraints.append(cp.abs(J @ x) <= np.pi/180*config.angle_limit),

    if config.generation_limit: # generation growth constraint

        constraints.append(c <= config.generation_limit)

    else: # only allow generation to be added at non-PQ busses

        k = bus[bus.BUS_TYPE == 1].index.astype(int).tolist()  # PQ bus list
        constraints.append(c[k] == 0)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    try:

        problem.solve(**config.cvx_solver)        
        value = problem.value
        status = problem.status.startswith("optimal")
        generators = c.value
        capacitors = np.clip(d.value,a_min=0,a_max=None)
        condensers = np.clip(-d.value,a_min=0,a_max=None)
        voltages = y.value * np.exp(x.value*1j)
        error = None

    except Exception as err:

        value = None
        status = False
        generators = None
        capacitors = None
        condensers = None
        voltages = None
        error = str(err)

    result = {x:eval(y) for x,y in config.results.items()}
    return result

if __name__ == "__main__":

    import os, sys
    import time

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

    tests = sorted([x[4:-3] for x in os.listdir("../test") if x.startswith("case") and x.endswith(".py")])
    errors = 0
    warnings = 0
    sz = max([len(x) for x in tests])
    print("Case"," "*(sz+2),f"{'Result':^20s}",f"{'Time':>8s}")
    print("-"*(sz+7),"-"*20,"--------")
    for caseid in tests:

        case = f"../test/case{caseid}.py"
        print(f"case{caseid}.py"," "*(sz-len(caseid)),end="",flush=True)

        test = PPModel(os.path.splitext(os.path.basename(case))[0],case=case)
        tic = time.time()
        result = runosp(test)
        toc = time.time()
        if result["status"] == False:
            errors += 1
        elif result["problem"].status != "optimal":
            warnings += 1
        print(f"{str(result['problem'].status):^20s}",f"{toc-tic:8.3f}")
        if not result["status"] and error_dump:
            print(flush=True)
            print(flush=True,file=sys.stderr)
            test.print()
            print(flush=True)
            print(flush=True,file=sys.stderr)
            runosp(test,{"cvx_solver":{"verbose":True}})
            if error_stop:
                break

    if errors > 0:
        print(f"{errors} error{'s' if errors > 1 else ''} found")
        if error_stop:
            print("Stopping on error")
            sys.exit(1)
    if warnings > 0:
        print(f"{warnings} warning{'s' if warnings > 1 else ''} found")
    if errors == 0 and warnings == 0:
        print("No error found")
    sys.exit(errors)
