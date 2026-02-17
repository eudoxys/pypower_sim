r"""Optimal sizing/placement solver

This function solves the optimal sizing and placement (OSP) problem for a network.
The OSP problem is to find the minimum cost allocation of capacitors, condensers,
and generators that satisfies the load, including the reserve margin.

# Costs

If the model contains `costs` data associated with each bus, then individual
bus costs are used to compute the total cost.  If no `costs` data is
available, then defaults costs for generators, capacitors, synchronous
condensers can be given as an optional argument.

The bus `costs` data table contains the following data

- `generator_power`: the cost of adding real power capacity to a generator
  in \$/MW.

- `generator_reactive`: the cost of adding reactive power capacity to a
  generator \$/MVAr.

- `capacitor`: the cost of adding a capacitor reactive power capacity
  in \$/MVAr.

- `condenser`: the cost of adding a synchronous condenser reactive power capacity $/MVAr.

Note that capacitors can only inject reactive power, whereas synchronous
condenser can both inject and absorb reactive power.

# Methodology

Optimal sizing and placement (OSP) seeks to identify the lowest cost
configuration of generators, capacitors, and condensers that guarantees
demand can be met. The OSP is the solution to the following convex
optimization problem for a network having $N$ busses and $M$ branches.

$\underset{x,y,g,h,c}{\min} \qquad P \sqrt{g^2+h^2} + Q |h| + \frac12 (R+S) |c| + \frac12(R-S) c$
    
$\textrm{subject to}$

$\qquad \Re(G) x - g + c + (\Re (D))(1+E) = 0 \qquad \textrm{real power flow balance}$
$\qquad \Im(G) y - h - c + (\Im (D))(1+E) = 0 \qquad \textrm{reactive power flow balance}$
$\qquad x_{ref} = 0 \qquad \textrm{reference bus voltage angle is always 0}$
$\qquad y_{ref} = 1 \qquad \textrm{reference bus voltage magnitude is always 1}$
$\qquad |y-1| \le 0.05 \qquad \textrm{bus voltages within 5\% of nominal}$
$\qquad |Ix| \le F \qquad \textrm{line flow constraints}$
$\qquad |h| < 0.2 g \qquad \textrm{constrain reactive power generation to 20\% of real power}$


with variables

- $c \in \mathbb{R}^N$ is the capacitor/condenser size
- $g \in \mathbb{R}^N$ is the generator real power capacity,
- $h \in \mathbb{R}^N$ is the generator reactive power capacity, and
- $x \in \mathbb{R}^N$ is the voltage angle,
- $y \in \mathbb{R}^N$ is the voltage magnitude,

and parameters

- $D \in \mathbb{C}^N$ is the total demand,
- $E \in \mathbb{R}$ is the demand safety margin,
- $F \in \mathbb{R}^M$ is the line flow constraints,
- $G \in \mathbb{C}^{N \times N}$ is the graph Laplacian,
- $I \in \mathbb{R}^{M \times N}$ is the graph incidence matrix,
- $P \in \mathbb{R}^N$ is the generation energy price,
- $Q \in \mathbb{R}^N$ is the price of reactive power control (not including energy),
- $R \in \mathbb{R}^N$ is the price of installing capacitors, and
- $S \in \mathbb{R}^N$ is the price of installing synchronous condensers

An additional constraint can be imposed if you wish to limit where generators
can be placed to only locations where generators exist, and limit how much
generation is located there to some factor, say 2x existing existing
capacity, e.g.,

$\qquad 0 \le \sqrt{g^2+h^2} \le 2 \Re H \qquad \textrm{generation total power constraint}$

where

- $H \in \mathbb{R}^N$ is the intalled generation capacity.

However, this constraint can result in an infeasible problem, and should not
be included if feasibility must be assured.

# Troubleshooting

The following optimizer results may be observed:

## Optimal

A complementary (primal and dual) solution has been found. The primal and dual
variables are replaced with their computed values, and the objective value of
the problem returned.

## Infeasible

The problem is infeasible as a result of an unbounded direction. The values of
the variables are filled with `NaN`, and the objective value is set to
$+\infty$ for minimizations and feasibility problems, and $-\infty$ for
maximizations.

## Unbounded

The solver has determined that the problem is unbounded. The objective value
is set to $-\infty$ for minimizations, and $+\infty$ for maximizations. The
values of any dual variables are replaced with `NaN`, as the dual problem is
in fact infeasible.

For unbounded problems, CVX stores an unbounded direction into the problem
variables. This is is a direction along which the feasible set is unbounded,
and the optimal value approaches $\pm\infty$. 

## Inaccurate

The solution may be inaccurate for the following reasons.

### Optimal/Unbounded/Infeasible

These three status values indicate that the solver was unable to make a
determination to within the default numerical tolerance. However, it
determined that the results obtained satisfied a relaxed tolerance level and
therefore may still be suitable for further use. If this occurs, you should
test the validity of the computed solution before using it in further
calculations.  

### Approximation

This status value indicates that the voltage angle assumption required by
the **Settings** $\rightarrow$ **Voltage angle accuracy limit** has been
exceeded by one or more voltages in the solution. Exceeding this limit
implies that the solution is inaccurate because the error in the
approximation $\sin(x) \approx x$ used by the fast-decoupled powerflow
constraints is unacceptably large. Exceptionally large angles, e.g., in
excess of 45$^\circ$ will result in potentially wildly inaccurate results. In
general, the simplest solution is to add busses on branches over which large
angles are observed.

## Failed

The solver failed to make sufficient progress towards a solution, even to
within the relaxed tolerance setting. The objective values and primal and
dual variables are filled with `NaN`. This result usually arises from
numerical problems within the model itself.

## Overdetermined

The presolver has determined that the problem has more equality constraints
than variables, which means that the coefficient matrix of the equality
constraints is singular. In practice, such problems are often, but not
always, infeasible. Unfortunately, solvers typically cannot handle such
problems, so a precise conclusion cannot be reached. This result usually
arises from numerical problems within the model itself.

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

def runosp(
    model:TypeVar('pypower_sim.ppmodel.PPModel'),
    costs:dict[str,float]|None=None,
    limits:dict[str,float]|None=None,
    update:bool=True,
    ) -> dict:
    """Solve the optimal sizing/placement problem

    # Arguments

    - `model`: a `pypower_sim.ppmodel.PPModel` object

    - `costs`: generation, capacitor, and condenser default costs

    - `limits`: generation capacity expansion limits, if any

    - `update`: specify whether to update the original model data with the solution

    # Returns

    - `dict`: result case data updated with the following added

      - `"success"` (`bool`): indicate whether a solution was found

      - `"reason"` (`str`|`None`): reason for `success==0` otherwise `None`

      - `"additions`" (`dict[numpy.ndarray]`): generation, capacitor, and
        condenser additions by bus

      - `"lagrange_multipliers"` (`dict[numpy.ndarray`): Lagrange multipliers for each inequality
        constraint, e.g., `voltage`, `flow`, `reactive_power`, and
        optionally `real_power` if `limits` are enabled.
    """

    # get graph analysis data
    bus = model.get_data("bus")
    branch = model.get_data("branch")
    gen = model.get_data("gen")
    gencost = model.get_data("gencost")
    construction = model.get_data("construction")

    # TODO: support adjustments to these
    margin = 0.2
    ref = 0
    voltage_limit = 0.05
    reactive_power_constraint = 0.02
    generator_expansion_limit = sp.sparse.coo_array((gen["PMAX"].values,(gen["GEN_BUS"].values,[1]*len(gen)))).T
    min_power_ratio = np.full(len(bus),0.2)

    graph = model._graph()
    
    N = len(bus)
    G = graph.laplacian(weighted=True,complex_flows=True).todense() # weighted graph Laplacian
    D = array([complex(*z) for z in bus[["PD","QD"]].values],ndmin=1) # demand
    I = graph.incidence(weighted=True,complex_flows=True).todense().T # network incidence matrix
    F = array(branch["RATE_A"].values,ndmin=1) # line ratings
    i,j = gen["GEN_BUS"].values,array([0]*len(gen)) # generator bus index
    v = array([complex(x,y) for x,y in gen[["PMAX","QMAX"]].values],ndmin=1) # generator capacities
    S = sp.sparse.coo_array((v,(i,j))).todense() # generators and synchronous condensers
    M = np.absolute(S) # current capacity
    C = np.clip(bus["BS"].values,a_min=0,a_max=None) # static capacitors (positive BS)
    R = np.clip(bus["BS"].values,a_min=None,a_max=0) # static condensers (negative BS)
    H = (M==0).astype(int) # flag busses where no capacity upgrade is possible
    U = 3*np.sum(np.absolute(D)) # upper bound

    x = cp.Variable(N,name='x') # bus voltage angles
    y = cp.Variable(N,name='y') # bus voltage magnitudes
    g = cp.Variable(N,name='g') # generation real power dispatch
    h = cp.Variable(N,name='h') # generation/synchronous condenser reactive power dispatch
    c = cp.Variable(N,name='c') # capacitor/condenser settings
    m = cp.Variable(N,name='m')
    
    puS = model.case["baseMVA"]
    s = cp.abs(g+h*1j) # apparent power at each bus
    i,j = construction["BUS_I"].values,[0]*len(construction)
    gen_cost = sp.sparse.coo_array((construction["GENERATOR"].values.astype(complex),(i,j))).todense().T[0]
    con_cost = sp.sparse.coo_array((construction["CONDENSER"].values,(i,j))).todense().T[0]
    rea_cost = sp.sparse.coo_array((construction["REACTOR"].values,(i,j))).todense().T[0]
    cap_cost = sp.sparse.coo_array((construction["CAPACITOR"].values,(i,j))).todense().T[0]

    costs = \
        cp.abs(gen_cost) @ cp.abs(g) \
      + cp.abs(gen_cost.imag) @ cp.abs(h) \
      + (cap_cost+con_cost)/2 @ cp.abs(c) \
      + (cap_cost-con_cost)/2 @ c

    constraints = [
        G.real @ x - g - S.real + c + C + R + D.real*(1+margin) == 0,  # KCL/KVL real power laws
        G.imag @ y - h - S.imag - c - C - R + D.imag*(1+margin) == 0,  # KCL/KVL reactive power laws
        x[ref-1] == 0,  # swing bus voltage angle always 0
        y[ref-1] == 1,  # swing bus voltage magnitude is always 1
        cp.abs(y - 1) <= voltage_limit,  # limit voltage magnitude to 5% deviation
        cp.abs(I.real@x + I.imag@y) <= F, 
        g >= 0, # generation must be positive
        cp.abs(h) <= reactive_power_constraint*g, # limit how much reactive power a generator can produce
        s <= U*m, # limit apparent power to upper bounds
        m >= 0, 
        m <= 1
        ]

    objective = cp.Minimize(costs)  # minimum cost (generation + demand response)

    if not generator_expansion_limit is None: #TODO: #change this to if generator_expansion_limit
        # limit where and how much generation can be added
        constraints.append(cp.abs(g+h*1j) <= generator_expansion_limit*cp.abs(S))

    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=False)
    result = {
      # "problem": problem.get_problem_data(solver=problem.solver_stats.solver_name),
      "cost": problem.value,
      "status": problem.status,
      }
    # setup problem

    # contruct result
    # result["success"] = 0
    # result["reason"] = "Not implemented yet"
    return result

if __name__ == "__main__":

    from pypower_sim.ppmodel import PPModel
    wheatstone = {
        "version": '2',
        "baseMVA": 100.0,
        "bus": array([
            [0, 3, 50,  30.99,  0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
            [1, 1, 170, 105.35, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
            [2, 1, 200, 123.94, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
            [3, 2, 80,  49.58,  0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
            ]),
        "gen": array([
            [0, 0,   0, 100, -100, 1,    100, 1, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 318, 0, 100, -100, 1.02, 100, 1, 318, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]),
        "branch": array([
            [0, 1, 0.01008, 0.0504, 0.1025, 250, 250, 250, 0, 0, 1, -360, 360],
            [0, 2, 0.00744, 0.0372, 0.0775, 250, 250, 250, 0, 0, 1, -360, 360],
            [1, 2, 0.00744, 0.0372, 0.0775, 250, 250, 250, 0, 0, 0, -360, 360],
            [1, 3, 0.00744, 0.0372, 0.0775, 250, 250, 250, 0, 0, 1, -360, 360],
            [2, 3, 0.01272, 0.0636, 0.1275, 250, 250, 250, 0, 0, 1, -360, 360],
            ]),
        "gencost": array([
            [2, 0.0, 0.0, 3, 0.04,20.0,0.0],
            [2, 0.0, 0.0, 3, 0.25,20.0,0.0],
          ]),
        "construction": array([ # construction costs ($/MW)
          # bus_i,generator,condenser,reactor,capacitor
          [1,0,0,0,1e4],
          [2,0,0,0,1e4],
          [3,1e6,1e5,0,0],
          ])
        }

    print(runosp(PPModel(case=wheatstone)))
