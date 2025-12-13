import random
from collections import defaultdict
from gurobipy import Model, GRB, quicksum

# --------------------- Toy instance generator ---------------------
def generate_toy_instance(S=6, K=2, P=3, G=1, J_options=3, T_per_stage=4, seed=1):
    random.seed(seed)
    data = {}
    data["S"] = S
    data["K"] = list(range(1, K+1))
    data["P"] = list(range(1, P+1))
    data["G"] = list(range(1, G+1))
    data["T_s"] = {s: list(range(1, T_per_stage+1)) for s in range(1, S+1)}
    data["J"] = { (s,k): list(range(1, J_options+1)) for s in range(1, S+1) for k in data["K"] }

    V = {}
    W = {}
    H = {}
    for s in range(1, S+1):
        for k in data["K"]:
            for j in data["J"][(s,k)]:
                for t in data["T_s"][s]:
                    # randomly decide visibilities (toy)
                    if random.random() < 0.5:
                        p = random.choice(data["P"])
                        V[(s,k,t,j,p)] = 1
                    if random.random() < 0.25:
                        g = random.choice(data["G"])
                        W[(s,k,t,j,g)] = 1
                    H[(s,k,t,j)] = 1 if random.random() < 0.9 else 0
    data["V"] = V
    data["W"] = W
    data["H"] = H

    data["Dobs"] = 5.0
    data["Dcomm"] = 4.0
    data["Bobs"] = 1.0
    data["Bcomm"] = 0.8
    data["Bcharge"] = 2.0
    data["Btime"] = 0.2
    data["Dmin"] = 0.0
    data["Dmax"] = 500.0
    data["Bmin"] = 0.0
    data["Bmax"] = 1000.0


    c = {}
    for s in range(1, S+1):
        for k in data["K"]:
            prev_list = data["J"][(s-1,k)] if s>1 else data["J"][(1,k)]
            for i in prev_list:
                for j in data["J"][(s,k)]:
                    c[(s,k,i,j)] = 0.0 if i==j else random.uniform(0.01, 0.3)
    data["c"] = c
    data["c_k_max"] = {k: 2.0 for k in data["K"]}

    data["d_init"] = {k: data["Dmin"] for k in data["K"]}
    data["b_init"] = {k: data["Bmax"] for k in data["K"]}

    data["C"] = 2.0  
    return data


def solve_rhp_subproblem_gurobi(data, s, L, J_tilde_prev, time_limit=None, verbose=False):

    model = Model(f"RHP_s{s}_L{L}")
    model.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    S = data["S"]
    K_list = data["K"]
    P_list = data["P"]
    G_list = data["G"]
    C = data["C"]
    Dobs = data["Dobs"]
    Dcomm = data["Dcomm"]

    L_end = min(s + L - 1, S)
    Lset = list(range(s, L_end + 1))
    T_s = {ell: data["T_s"][ell] for ell in Lset}

    
    x = {}   
    y = {}   
    q = {}   
    h = {}   
    d = {}   
    b = {}   

    
    for ell in Lset:
        for k in K_list:
            prev_i_list = [J_tilde_prev[k]] if ell == s else data["J"][(ell-1, k)]
            for i in prev_i_list:
                for j in data["J"][(ell, k)]:
                    x[(ell, k, i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{ell}_{k}_{i}_{j}")
    
    for ell in Lset:
        for k in K_list:
            for t in T_s[ell]:
                d[(ell, k, t)] = model.addVar(lb=data["Dmin"], ub=data["Dmax"], vtype=GRB.CONTINUOUS, name=f"d_{ell}_{k}_{t}")
                b[(ell, k, t)] = model.addVar(lb=data["Bmin"], ub=data["Bmax"], vtype=GRB.CONTINUOUS, name=f"b_{ell}_{k}_{t}")
                h[(ell, k, t)] = model.addVar(vtype=GRB.BINARY, name=f"h_{ell}_{k}_{t}")
                for p in P_list:
                    y[(ell, k, t, p)] = model.addVar(vtype=GRB.BINARY, name=f"y_{ell}_{k}_{t}_{p}")
                for gidx in G_list:
                    q[(ell, k, t, gidx)] = model.addVar(vtype=GRB.BINARY, name=f"q_{ell}_{k}_{t}_{gidx}")

    model.update()

    obj_terms = []
    for key, val in q.items():
        obj_terms.append(C * val)
    for key, val in y.items():
        obj_terms.append(val)
    model.setObjective(quicksum(obj_terms), GRB.MAXIMIZE)

    for k in K_list:
        i = J_tilde_prev[k]
        expr = quicksum(x[(s, k, i, j)] for j in data["J"][(s, k)] if (s, k, i, j) in x)
        model.addConstr(expr == 1, name=f"cont_first_{s}_{k}")
        
    for ell in Lset:
        if ell == max(Lset):
            continue
        for k in K_list:
            for i in data["J"][(ell, k)]:
                left_terms = [ x[(ell+1, k, i, j)] for j in data["J"][(ell+1, k)] if (ell+1, k, i, j) in x ]
                prev_list = [J_tilde_prev[k]] if ell == s else data["J"][(ell-1, k)]
                right_terms = [ x[(ell, k, jprev, i)] for jprev in prev_list if (ell, k, jprev, i) in x ]
                model.addConstr(quicksum(left_terms) - quicksum(right_terms) == 0, name=f"cont_chain_{ell}_{k}_{i}")

    for k in K_list:
        expr_terms = []
        for (ell_, kk, i, j) in x.keys():
            if kk != k:
                continue
            cost = data["c"].get((ell_, kk, i, j), 0.0)
            expr_terms.append(cost * x[(ell_, kk, i, j)])
        model.addConstr(quicksum(expr_terms) <= data["c_k_max"][k], name=f"transfer_budget_{k}")

    for (ell, k, t, p), vary in y.items():
        lhs = []
        prev_list = [J_tilde_prev[k]] if ell == s else data["J"][(ell-1, k)]
        for i in prev_list:
            for j in data["J"][(ell, k)]:
                if data["V"].get((ell, k, t, j, p), 0) == 1 and (ell, k, i, j) in x:
                    lhs.append(x[(ell, k, i, j)])
        if not lhs:
            model.addConstr(vary == 0, name=f"no_vis_y_{ell}_{k}_{t}_{p}")
        else:
            model.addConstr(quicksum(lhs) >= vary, name=f"vtw_y_{ell}_{k}_{t}_{p}")

    for (ell, k, t, gidx), varq in q.items():
        lhs = []
        prev_list = [J_tilde_prev[k]] if ell == s else data["J"][(ell-1, k)]
        for i in prev_list:
            for j in data["J"][(ell, k)]:
                if data["W"].get((ell, k, t, j, gidx), 0) == 1 and (ell, k, i, j) in x:
                    lhs.append(x[(ell, k, i, j)])
        if not lhs:
            model.addConstr(varq == 0, name=f"no_vis_q_{ell}_{k}_{t}_{gidx}")
        else:
            model.addConstr(quicksum(lhs) >= varq, name=f"vtw_q_{ell}_{k}_{t}_{gidx}")

    for (ell, k, t), varh in h.items():
        lhs = []
        prev_list = [J_tilde_prev[k]] if ell == s else data["J"][(ell-1, k)]
        for i in prev_list:
            for j in data["J"][(ell, k)]:
                if data["H"].get((ell, k, t, j), 0) >= 0.5 and (ell, k, i, j) in x:
                    lhs.append(x[(ell, k, i, j)])
        if not lhs:
            model.addConstr(varh == 0, name=f"no_vis_h_{ell}_{k}_{t}")
        else:
            model.addConstr(quicksum(lhs) >= varh, name=f"vtw_h_{ell}_{k}_{t}")

    for ell in Lset:
        for k in K_list:
            for t in T_s[ell]:
                expr = quicksum(y[(ell, k, t, p)] for p in P_list) + quicksum(q[(ell, k, t, gidx)] for gidx in G_list) + h[(ell, k, t)]
                model.addConstr(expr <= 1, name=f"at_most_one_{ell}_{k}_{t}")

    for ell in Lset:
        Tlist = T_s[ell]
        for k in K_list:
            for t in Tlist:
                if t < max(Tlist):
                    model.addConstr(d[(ell, k, t+1)] == d[(ell, k, t)] + Dobs * quicksum(y[(ell, k, t, p)] for p in P_list) - Dcomm * quicksum(q[(ell, k, t, gidx)] for gidx in G_list),
                                    name=f"data_update_{ell}_{k}_{t}")
    for ell in Lset:
        if ell == max(Lset):
            continue
        Tend = max(T_s[ell])
        for k in K_list:
            model.addConstr(d[(ell+1, k, 1)] == d[(ell, k, Tend)] + Dobs * quicksum(y[(ell, k, Tend, p)] for p in P_list) - Dcomm * quicksum(q[(ell, k, Tend, gidx)] for gidx in G_list),
                            name=f"data_stage_bound_{ell}_{k}")


    for (ell, k, t), dv in d.items():
        model.addConstr(dv + Dobs * quicksum(y[(ell, k, t, p)] for p in P_list) <= data["Dmax"], name=f"data_upper_{ell}_{k}_{t}")
        model.addConstr(dv - Dcomm * quicksum(q[(ell, k, t, gidx)] for gidx in G_list) >= data["Dmin"], name=f"data_lower_{ell}_{k}_{t}")

    for ell in Lset:
        Tlist = T_s[ell]
        for k in K_list:
            for t in Tlist:
                if t < max(Tlist):
                    model.addConstr(b[(ell, k, t+1)] == b[(ell, k, t)] + data["Bcharge"] * h[(ell, k, t)] - data["Bobs"] * quicksum(y[(ell, k, t, p)] for p in P_list) - data["Bcomm"] * quicksum(q[(ell, k, t, gidx)] for gidx in G_list) - data["Btime"],
                                    name=f"batt_update_{ell}_{k}_{t}")

    for ell in Lset:
        if ell == max(Lset):
            continue
        Tend = max(T_s[ell])
        for k in K_list:
            model.addConstr(b[(ell+1, k, 1)] == b[(ell, k, Tend)] + data["Bcharge"] * h[(ell, k, Tend)] - data["Bobs"] * quicksum(y[(ell, k, Tend, p)] for p in P_list) - data["Bcomm"] * quicksum(q[(ell, k, Tend, gidx)] for gidx in G_list) - data["Btime"],
                            name=f"batt_stage_bound_{ell}_{k}")

    for (ell, k, t), bv in b.items():
        model.addConstr(bv + data["Bcharge"] * h[(ell, k, t)] <= data["Bmax"], name=f"batt_upper_{ell}_{k}_{t}")
        model.addConstr(bv - data["Bobs"] * quicksum(y[(ell, k, t, p)] for p in P_list) - data["Bcomm"] * quicksum(q[(ell, k, t, gidx)] for gidx in G_list) - data["Btime"] >= data["Bmin"],
                        name=f"batt_lower_{ell}_{k}_{t}")

    for k in K_list:
        model.addConstr(d[(s, k, 1)] == data["d_init"][k], name=f"init_d_{k}")
        model.addConstr(b[(s, k, 1)] == data["b_init"][k], name=f"init_b_{k}")

    model.update()
    model.optimize()

    return model, x, y, q, h, d, b

def rolling_horizon_gurobi(data, L=1, time_limit_per_sub=30, verbose=False):
    S = data["S"]
    K_list = data["K"]

    J_tilde_prev = {k: data["J"][(1, k)][0] for k in K_list}

    x_tilde = {}
    y_tilde = {}
    q_tilde = {}
    h_tilde = {}
    d_tilde = {}
    b_tilde = {}
    z_history = []

    for s in range(1, S - L + 2):  
        if verbose:
            print(f"\n--- Solving RHP(s={s}, L={L}) ---")
        model, xvars, yvars, qvars, hvars, dvars, bvars = solve_rhp_subproblem_gurobi(data, s, L, J_tilde_prev, time_limit=time_limit_per_sub, verbose=verbose)

        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            if verbose:
                print("Solver status:", model.Status)
            z_history.append(None)
            continue

        zval = model.ObjVal if model.ObjVal is not None else None
        z_history.append(zval)
        if verbose:
            print("z_s:", zval)

        for (ell, k, i, j), var in list(xvars.items()):
            if ell == s:
                val = int(round(var.X)) if hasattr(var, "X") else 0
                if val == 1:
                    x_tilde[(s, k)] = (i, j)
                    J_tilde_prev[k] = j

        for (ell, k, t, p), var in list(yvars.items()):
            if ell == s:
                if int(round(var.X)) == 1:
                    y_tilde.setdefault((s, k), []).append((t, p))
        for (ell, k, t, g), var in list(qvars.items()):
            if ell == s:
                if int(round(var.X)) == 1:
                    q_tilde.setdefault((s, k), []).append((t, g))
        for (ell, k, t), var in list(hvars.items()):
            if ell == s:
                if int(round(var.X)) == 1:
                    h_tilde.setdefault((s, k), []).append(t)

        for (ell, k, t), var in list(dvars.items()):
            if ell == s:
                d_tilde[(s, k, t)] = var.X
        for (ell, k, t), var in list(bvars.items()):
            if ell == s:
                b_tilde[(s, k, t)] = var.X

        Tend = max(data["T_s"][s])
        for k in K_list:
            d_Tend = dvars[(s, k, Tend)].X
            y_Tend = sum(int(round(yvars[(s, k, Tend, p)].X)) for p in data["P"] if (s, k, Tend, p) in yvars)
            q_Tend = sum(int(round(qvars[(s, k, Tend, g)].X)) for g in data["G"] if (s, k, Tend, g) in qvars)
            data["d_init"][k] = d_Tend + data["Dobs"] * y_Tend - data["Dcomm"] * q_Tend
            b_Tend = bvars[(s, k, Tend)].X
            h_Tend = int(round(hvars[(s, k, Tend)].X)) if (s, k, Tend) in hvars else 0
            data["b_init"][k] = b_Tend + data["Bcharge"] * h_Tend - data["Bobs"] * y_Tend - data["Bcomm"] * q_Tend - data["Btime"]

            chosen = x_tilde.get((s, k), None)
            if chosen:
                i, j = chosen
                cost = data["c"].get((s, k, i, j), 0.0)
                data["c_k_max"][k] = max(0.0, data["c_k_max"][k] - cost)

    return {
        "x_tilde": x_tilde,
        "y_tilde": y_tilde,
        "q_tilde": q_tilde,
        "h_tilde": h_tilde,
        "d_tilde": d_tilde,
        "b_tilde": b_tilde,
        "z_history": z_history
    }

if __name__ == "__main__":
    inst = generate_toy_instance(S=6, K=2, P=3, G=1, J_options=3, T_per_stage=3, seed=42)
    print("Running rolling horizon (toy) with Gurobi. Ensure gurobipy is installed and licensed.")
    res = rolling_horizon_gurobi(inst, L=1, time_limit_per_sub=15, verbose=False)
    print("\n-- Summary --")
    print("Objective history (per subproblem):", res["z_history"])
    print("First-stage orbital choices x_tilde:", res["x_tilde"])
    print("First-stage observations y_tilde:", res["y_tilde"])
    print("First-stage downlinks q_tilde:", res["q_tilde"])
    print("First-stage charges h_tilde:", res["h_tilde"])
