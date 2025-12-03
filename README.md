[![validate](https://github.com/eudoxys/pypower_sim/actions/workflows/validate.yaml/badge.svg)](https://github.com/eudoxys/pypower_sim/actions/workflows/validate.yaml)

# pypower_sim

PyPOWER timeseries simulation

## Installation

    python3 -m venv .venv
    . .venv/bin/activate
    pip install git+https://github.com/eudoxys.com/pypower_sim

## Running Examples

    cd examples
    python3

    from pypower_sim import PPModel
    from wecc240 import wecc240
    model = PPModel(case=wecc240)

    from pypower_sim import PPSolver
    solver=PPSolver(model)
    solver.solve_opf()
    solver.solve_opf(use_acopf=True)
    solver.solve_pf(with_result=True)

