import gurobipy as gp
from gurobipy import GRB

#真の順問題を解く(ノイズなし)
def solve_forward(cfg,Q,theta_i,A,b,Sigma_inv):
    model = gp.Model()
    model.setParam("OutputFlag",0)
    
    a = model.addVars(cfg.D,lb=0)
    
    w1 = sum(Q[0,h] * theta_i[h] for h in range(cfg.H))
    w2 = sum(Q[1,h] * theta_i[h] for h in range(cfg.H))
    
    obj = gp.quicksum(w1 * a[d] * a[d] for d in range(cfg.D)) + gp.quicksum(w2 * Sigma_inv[d1][d2] * a[d1] * a[d2] for d1 in range(cfg.D) for d2 in range(cfg.D))
    
    
    
    
    model.setObjective(obj,GRB.MINIMIZE)
    
    for m in range(cfg.M):
        model.addConstr(
            gp.quicksum(A[m,d]*a[d] for d in range(cfg.D)) <= b[m]
        )
    
    model.optimize()
    return [a[d].X for d in range(cfg.D)]

