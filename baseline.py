import gurobipy as gp
from gurobipy import GRB

def solve_euclid(cfg,A,b):
    model = gp.Model()
    model.setParam("OutputFlag",0)
    
    a = model.addVars(cfg.D,lb=0)
    

    
    obj = gp.quicksum(
        a[d] * a[d]
        for d in range(cfg.D)
    )
    
    model.setObjective(obj,GRB.MINIMIZE)
    
    for m in range(cfg.M):
        model.addConstr(
            gp.quicksum(A[m,d]*a[d] for d in range(cfg.D)) <= b[m]
        )
    
    model.optimize()
    return [a[d].X for d in range(cfg.D)]


def solve_maha(cfg,A,b,Sigma_inv):
    model = gp.Model()
    model.setParam("OutputFlag",0)
    
    a = model.addVars(cfg.D,lb=0)
    
    
    obj = gp.quicksum(Sigma_inv[d1][d2] * a[d1] * a[d2] for d1 in range(cfg.D) for d2 in range(cfg.D))
    
    model.setObjective(obj,GRB.MINIMIZE)
    
    for m in range(cfg.M):
        model.addConstr(
            gp.quicksum(A[m,d]*a[d] for d in range(cfg.D)) <= b[m]
        )
    
    model.optimize()
    return [a[d].X for d in range(cfg.D)]