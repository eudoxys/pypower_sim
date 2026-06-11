"""Capacity violation detector"""
from collections import namedtuple
from warnings import warn

import numpy as np

from pypower import idx_bus as bus
from pypower import idx_brch as branch
from pypower import idx_gen as gen
from pypower import idx_cost as cost
from pypower import idx_dcline
dcline = namedtuple("dcline",idx_dcline.c.keys())(**idx_dcline.c)

def violations(data, 
    precision:float=-1, # rounding on data before checking
    error:float=0.02, # error margin on tests
    ) -> int|str|dict:
    """Enumerate violations in case

    Arguments
    ---------

    - `data`: case data

    - `precision`: floating point precision to use for data

    - `error`: fraction error tolerance for data checks

    - `formatter`: formatting of return value

    Returns
    -------

    `int`: count of violations (`formatter`==`"counter"`)

    `str`: table of violations (`formatter`==`"table"`)

    `dict`: violations data (`formatter`==`None`)

    *varies*: whatever `formatter` return if callable
    """
    result = {"bus": [], "gen": [], "branch": []}
    if "dcline" in data:
        result["dcline"] = []

    # check busses
    if "bus" in data:
        for n, v in enumerate(
            data["bus"][:, (bus.VM, bus.VA, bus.VMIN, bus.VMAX)].round(precision)
        ):
            VM, VA, VMIN, VMAX = map(float, v)
            if not VMIN*(1-error) <= VM <= VMAX*(1+error):
                result["bus"].append((n, f"{VM=} pu.V outside ({VMIN=},{VMAX=})"))

    # check generators
    if "gen" in data:
        for n, g in enumerate(data["gen"][
                :, (gen.GEN_STATUS,gen.PG, gen.QG, gen.PMIN, gen.PMAX, gen.QMIN, gen.QMAX)
            ].round(precision)
        ):
            STATUS, PG, QG, PMIN, PMAX, QMIN, QMAX = map(float, g)
            if STATUS == 0 or PG == 0:
                continue
            if PMIN < PMAX and not PMIN*(1-error) <= PG <= PMAX*(1+error):
                result["gen"].append((n, f"{PG=} MW outside ({PMIN=},{PMAX=})"))
            if QMIN >= 0 and QMIN < QMAX and not QMIN*(1-error) <= QG <= QMAX*(1+error):
                result["gen"].append((n, f"{QG=} MVAr outside ({QMIN=},{QMAX=})"))
            if QMIN < 0 and QMIN < QMAX and not QMIN*(1+error) <= QG <= QMAX*(1+error):
                result["gen"].append((n, f"{QG=} MVAr outside ({QMIN=},{QMAX=})"))
    
    # check branches
    if "branch" in data and data["branch"].shape[1] >= branch.PF:
        for n, b in enumerate(data["branch"][:,(branch.BR_STATUS,branch.PF,branch.RATE_A)]):
            STATUS, PF, RATE_A = map(float, np.abs(b))
            if STATUS == 1 and RATE_A > 0 and PF > RATE_A*(1+error):
                result["branch"].append((n, f"|PF|={PF:.1f} MVA outside (0,{RATE_A=:.1f})"))

    # check dclines
    if "dcline" in data and data["dcline"].shape[1]:
        for n,d in enumerate(data["dcline"][
                :,(dcline.BR_STATUS,dcline.PF,dcline.QF,dcline.PT,dcline.QT,dcline.PMIN,dcline.PMAX,
                    dcline.QMINF,dcline.QMAXF,dcline.QMINT,dcline.QMAXT)]):
            STATUS,PF,QF,PT,QT,PMIN,PMAX,QMINF,QMAXF,QMINT,QMAXT = map(float,d)
            if STATUS == 0 or PF == 0:
                continue
            if not PMIN*(1-error) <= PF <= PMAX*(1+error):
                result["dcline"].append((n,f"{PF=} outside ({PMIN=},{PMAX=})"))
            if not QMINF*(1+error) <= QF <= QMAXF*(1+error):
                result["dcline"].append((n,f"{QF=} outside ({QMINF=},{QMAXF=})"))
            if not QMINT*(1+error) <= QT <= QMAXT*(1+error):
                result["dcline"].append((n,f"{QT=} outside ({QMINT=},{QMAXT=})"))

    return result
