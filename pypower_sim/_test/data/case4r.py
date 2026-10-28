from numpy import array
def case4r():
    return {
        "version": '2',
        "baseMVA": 100.0,
        "bus": array([
            # BUS_I, BUS_TYPE,  PD, PQ, GS, BS, BUS_AREA, VM, VA, BASE_KV, ZONE, VMAX, VMIN 
            [     1,        3,   0,  0,  0,  0,        1,  1,  0,     230,    1,  1.1,  0.9],
            [     2,        1,   0,  0,  0,  0,        1,  1,  0,     230,    1,  1.1,  0.9],
            [     3,        1, 200, 10,  0,  0,        1,  1,  0,     230,    1,  1.1,  0.9],
            [     4,        2, 200, 10,  0,  0,        1,  1,  0,     230,    1,  1.1,  0.9],
            ]),
        "gen": array([
            # GEN_BUS, PG, QG, QMAX, QMIN,   VG, MBASE, GEN_STATUS, PMAX, PMIN, PC1, PC2, QC1MIN, QC1MAX, QC2MIN, QC1MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF
            [       1,  0,  0,  100, -100, 1.00,   100,          1,  400,    0,   0,   0,      0,      0,      0,      0,        0,       0,       0,      0,   0],
        ]),
        "branch": array([
            # F_BUS, T_BUS,  BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, BR_STATUS, ANGMIN, ANGMAX
            [     1,     2, 0.001, 0.02, 0.10,    400,    400,    500,   0,     0,         1,   -360,    360],
            [     2,     3, 0.002, 0.05, 0.08,    200,    200,    200,   0,     0,         1,   -360,    360],
            [     2,     4, 0.002, 0.05, 0.08,    200,    200,    200,   0,     0,         1,   -360,    360],
            [     3,     4, 0.003, 0.08, 0.12,     50,     50,     50,   0,     0,         1,   -360,    360],
            ]),
        "gencost": array([
            # MODEL, STARTUP, SHUTDOWN, N, COST0, ...
            [     2,     0.0,      0.0, 3,  0.04, 20.0, 0.0],
            ]),
        }

if __name__ == "__main__":

    from check_case import check_case
    name,case = [(x,eval(x)) for x in globals() if x.startswith("case")][0]
    check_case(name,case())
