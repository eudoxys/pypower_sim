def decoupled_acopf(
    data:dict,
    curtailment:float=None,
    setpoints:float|bool|None=None,
    smallangles:float|None=None,
    **kwargs,
    ) -> dict:
    """Solve decoupled optimal powerflow problem
    
    Arguments
    ---------

    - `data`: `pypower` case data

    - `curtailment`: cost of curtailment per-unit generation cost
      (None disables curtailment)

    - `setpoints`: constrain to voltages to gen.VG or set ref voltage(s)

    - `**options`: `cvxpy` solver options

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

      In addition the following are included when the problem is feasible:
      
      - `pf`: real power flow on branches
      - `qf`: reactive power flow on branches
      - `vm`: bus voltage magnitudes
      - `va`: bus voltage angles
      - `pg`: real power generation dispatch
      - `qg`: reactive power generation dispatch
      - `pc`: real power curtailment (if any)
      - `qc`: reactive power curtailment (if any)
    """
    
    tic = time()

    # default options
    options = cvx_options
    options.update(kwargs)

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
    bb = np.array(data["bus"])
    vl = cp.Constant(value=bb[:, [bus.VMIN]], name="vl") # voltage lower limit
    vu = cp.Constant(value=bb[:, [bus.VMAX]], name="vu") # voltage upper limit
    pd = cp.Parameter(shape=(N,1), value=bb[:, [bus.PD]] / puS, name="pd") # load real power
    qd = cp.Parameter(shape=(N,1), value=bb[:, [bus.QD]] / puS, name="qd") # load reactive power
    bi = {i: n for n, i in enumerate(bb[:, bus.BUS_I])}  # bus index (i is not necessarily reasonable)

    # branch parameters
    M = len(data["branch"])
    assert M > 0, "too few branches"
    br = data["branch"] # branch data

    f_bus = [bi[x] for x in br[:,branch.F_BUS]]
    t_bus = [bi[x] for x in br[:,branch.T_BUS]]
    
    tap = br[:,[branch.TAP]]
    tap[np.where(tap==0)] = 1.0 # non-zero is only for transformers, zero is powerline (unity tap)
    err = np.where(tap<0)[0]
    assert len(err) == 0, f"bus[{err},TAP] < 0"
    
    shift = br[:,[branch.SHIFT]] * np.pi / 180
    
    br_status = br[:,[branch.BR_STATUS]]
    err = np.where([x for x in br_status.flatten() if x not in [0,1]])[0]
    assert len(err)==0, f"bus[{err},BR_STATUS] value is not in [0,1]"
    
    br_x = br[:,[branch.BR_X]]
    err = np.where(br_x==0)[0]
    assert len(err) == 0, f"bus[{err},BR_X] <= 0"

    x = br_status/br_x/tap

    b = sp.sparse.coo_matrix((x.flatten(),(range(M),f_bus)),shape=(M,N)) \
        - sp.sparse.coo_matrix(((x+shift).flatten(),(range(M),t_bus)),shape=(M,N)) 
    b = cp.Constant(value=b, name="b") # line susceptances

    f = sp.sparse.coo_matrix((br_status.flatten(),(range(M),f_bus)),shape=(M,N)) \
        - sp.sparse.coo_matrix((br_status.flatten(),(range(M),t_bus)),shape=(M,N)) 
    f = cp.Constant(value=f.T, name="f") # line connections

    s = br[:,[branch.RATE_A]] / puS
    s[s==0] = 1e5 # zero ratings are unlimited
    s = cp.Parameter(shape=(M,1),value=s,name="s") # line flow limits

    # gen parameters
    K = len(data["gen"])
    assert K > 0, "too few generators"
    gg = np.array(data["gen"])
    gi = np.array([bi[n] for n in gg[:,gen.GEN_BUS]])
    vg = cp.Constant(value=gg[:,[gen.VG]], name="vg") # bus voltage setpoints
    pl = cp.Constant(value=gg[:,[gen.PMIN]]/puS, name="pl") # real power minimum
    pu = cp.Constant(value=gg[:,[gen.PMAX]]/puS, name="pu") # real power maximum
    ql = cp.Constant(value=gg[:,[gen.QMIN]]/puS, name="ql") # reactive power minimum
    qu = cp.Constant(value=gg[:,[gen.QMAX]]/puS, name="qu") # reactive power maximum
    g = sp.sparse.coo_matrix((np.ones(K),(list(range(K)),gi)),shape=(K,N)).T # sum generators to busses
    g = cp.Constant(value=g,name="g") # sum generators to busses

    # dc line parameters
    L = len(data["dcline"]) if "dcline" in data else 0
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
    pf = cp.Variable((M,1), name="pf")  # line real power flows
    qf = cp.Variable((M,1), name="qf")  # line reactive power flows
    vm = cp.Variable((N,1), name="|v|", nonneg=True)  # voltage magnitudes
    va = cp.Variable((N,1), name="𝞱")  # voltage angles
    pg = cp.Variable((K,1), name="pg", nonneg=True)  # generator real power dispatch
    qg = cp.Variable((K,1), name="qg")  # generator reactive power dispatch
    if not curtailment is None:
        pc = cp.Variable(shape=(N,1), name="pc", nonneg=True) # real power demand curtailment
        qc = cp.Variable(shape=(N,1), name="qc") # reactive power demand curtailment
    if L > 0:
        dpf = cp.Variable(shape=(L,1), name="dpf",nonneg=True) # dc line real power from
        dpt = cp.Variable(shape=(L,1), name="dpt",nonneg=True) # dc line real power to
        dqf = cp.Variable(shape=(L,1), name="dqf") # dc line reactive power from
        dqt = cp.Variable(shape=(L,1), name="dqt") # dc line reactive power to

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

    # reference busses
    ref = [n for n, bt in enumerate(data["bus"][:, bus.BUS_TYPE]) if bt == 3]
    assert len(ref) > 0, "no reference bus"

    # cost function
    cost = 0
    # cost = cp.sum(pg**2+qg**2) # TODO: replace with gencost if not just feasibility check
    if curtailment:
        cost += curtailment * cp.sum ( pc**2 + qc**2 ) # curtailment cost

    # constraints
    constraints = [

        # Feasible Set 2
        pf == b @ va, # Equation (1a)
        qf == b @ vm, # Equation (1b)

        # Feasible Set 4
        # pl <= pg, # non-zero not supported
        0 <= pg, pg <= pu, # Equation (3a)
        ql <= qg, qg <= qu, # Equation (3b)
        cp.abs(pf) <= s, # Equation (4a)
        cp.abs(qf) <= s,
        cp.abs(pf) + cp.abs(qf) <= 1.4 * s,
        vl <= vm, vm <= vu, # Equation (5a)
    ]

    # small angle assumption
    if not smallangles is None:
        constraints.append(cp.abs(va) <= smallangles)  # +/- 10 degrees for decoupling assumptions to be valid

    # bus voltage setpoints
    if isinstance(setpoints,float):
        constraints.append(vm[ref] == setpoints)
    elif setpoints is True:
        constraints.append(vm[gi] == vg)

    # curtailment
    if curtailment is None: # no curtailment allowed
        pn = pd # net load without curtailment
        qn = qd # net load without curtailment
    else: # curtailment is allowed
        pn = pd - pc # net load with curtailment
        qn = qd - qc # net load with curtailment
        constraints += [
            pc <= pd, # real power curtailment limits
            cp.minimum(qc,0) >= cp.minimum(qd,0), # reactive power curtailment lower limits
            cp.maximum(qc,0) <= cp.maximum(qd,0), # reactive power curtailment upper limits
        ]

    # dc lines
    if L == 0:
        constraints += [ # line flows without DC lines
            f @ pf + pn == g @ pg, # Equation (2a)
            f @ qf + qn == g @ qg, # Equation (2b)
            ]
    else:
        constraints += [ # line flows with DC lines
            f @ pf + pn + df @ dpf == g @ pg + dt @ dpt, # Equation (2a)
            f @ qf + qn + df @ dqf == g @ qg + dt @ dqt, # Equation (2b)
            dpt == cp.multiply(1-da,dpf) - db, # DC losses
            # dpmin <= dpf, # non-zero not supported
            0 <= dpf, dpf <= dpmax, # real power DC "to" injection limits
            dqminf <= dqf, dqf <= dqmaxf, # reacation power DC "from" injection limits
            dqmint <= dqt, dqt <= dqmaxf, # real power DC "to" injection limits
            ]

    # problem statement
    objective = cp.Minimize(cost)
    problem = cp.Problem(objective,constraints)
    problem.solve(**options)

    # solution results
    result = {
        "ok": False,
        "case": copy(data),
        "status": problem.status,
        "value": np.round(problem.value,4),
        "problem": problem,
        "objective": objective,
        "constraints": [str(x) for x in constraints],
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
        "warnings": warnings,
        "violations": {},
    }
    if L > 0:
        result["constants"].update({
            # "dl (pu.MVAr)": dl.value.T[0],
            "df (pu)": df.value.todense(), # TODO: remove todense
            "dt (pu)": dt.value.todense(), # TODO: remove todense
            "da (pu)": da.value.T[0],
            "db (pu.MW)": db.value.T[0],
            })
        result["parameters"].update({
            # "dm (pu)": dm.value,
            "dpmin (pu.MW)": dpmin.value.T[0],
            "dpmax (pu.MW)": dpmax.value.T[0],
            "dqminf (pu.MVAr)": dqminf.value.T[0],
            "dqmaxf (pu.MVAr)": dqmaxf.value.T[0],
            "dqmint (pu.MVAr)": dqmint.value.T[0],
            "dqmaxt (pu)": dqmaxt.value.T[0],
            })
    if problem.status == "optimal":
        result["variables"] = {
            "pf (pu.MW)": pf.value.T[0],
            "qf (pu.MVAr)": qf.value.T[0],
            "vm (pu.kV)": vm.value.T[0],
            "va (deg)": (va.value*180/np.pi).T[0],
            "pg (pu.MW)": pg.value.T[0],
            "qg (pu.MVAr)": qg.value.T[0],
        }
        if not curtailment is None:
            result["variables"].update({
                "pc (pu.MW)": pc.value.round(4).T[0],
                "qc (pu.MVAr)": qc.value.round(4).T[0],
                })
        if L > 0:
            result["variables"].update({
                "dpf (pu.MW)": dpf.value.round(4),
                "dqf (pu.MVAr)": dqf.value.round(4),
                "dpt (pu.MW)": dpt.value.round(4),
                "dqt (pu.MVAr)": dqt.value.round(4),
                })

        solution = copy(data)
        
        # update bus data
        bb = solution["bus"]
        bb[:,[bus.VA]] = ( va.value - va.value.T[0][ref[0]]) * 180 / np.pi
        bb[:,[bus.VM]] = vm.value
        if not curtailment is None:
            bb[:,[bus.PD]] = ( pd.value - pc.value ) * puS
            bb[:,[bus.QD]] = ( qd.value - qc.value ) * puS

        # update branch data
        bb = solution["branch"]
        bb[:,[branch.PF]] = pf.value * puS
        bb[:,[branch.QF]] = qf.value * puS
        bb[:,[branch.PT]] = 0
        bb[:,[branch.QT]] = 0

        # update gen
        gg = solution["gen"]
        gg[:,[gen.PG]] = pg.value * puS
        gg[:,[gen.QG]] = qg.value * puS
        # gg[:,[gen.PMIN]] = np.zeros(shape=(K,1))
        # gg[:,[gen.VG]] = vg.value

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

        result["solution"] = solution
        checks = violations(solution,formatter=dict)
        if checks:
            result["violations"] = checks
        result["ok"] = True

    toc = time()
    result["time"] = round(toc-tic,3)
    return result
