# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Power flow data for 4 bus, 2 gen case from Grainger & Stevenson.
"""

from numpy import array

def case4gsm():
    """Power flow data for 4 bus, 2 gen case from Grainger & Stevenson.
    Please see L{caseformat} for details on the case file format.

    This is the 4 bus example from pp. 337-338 of I{"Power System Analysis"},
    by John Grainger, Jr., William Stevenson, McGraw-Hill, 1994.

    @return: Power flow data for 4 bus, 2 gen case from Grainger & Stevenson.
    """
    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array([
        [0, 3, 5.0,  3.099,  0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [1, 1, 17.0, 10.535, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [2, 1, 20.0, 12.394, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [3, 2, 8.0,  4.958,  0, 0, 1, 1, 0, 230, 1, 1.1, 0.9]
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = array([
        [3, 318, 0, 100, -100, 1.02, 100, 1, 318, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0,   0, 100, -100, 1,    100, 1, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    ## branch data
    #fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([
        [0, 1, 0.01008, 0.0504, 0.1025, 250, 250, 250, 0, 0, 1, -360, 360],
        [0, 2, 0.00744, 0.0372, 0.0775, 250, 250, 250, 0, 0, 1, -360, 360],
        [1, 3, 0.00744, 0.0372, 0.0775, 250, 250, 250, 0, 0, 1, -360, 360],
        [2, 3, 0.01272, 0.0636, 0.1275, 250, 250, 250, 0, 0, 1, -360, 360]
    ])

    ppc["gencost"] = array([
            [2, 0.0, 0.0, 3, 0.04,20.0,0.0],
            [2, 0.0, 0.0, 3, 0.25,20.0,0.0],
            ])

    return ppc

if __name__ == "__main__":

    from check_case import check_case
    name,case = [(x,eval(x)) for x in globals() if x.startswith("case")][0]
    check_case(name,case())
