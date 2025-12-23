import gurobipy as gp
from gurobipy import GRB
import numpy as np

#参照点を用いた係数行列の推定方法
def inverse_proposed(cfg, data, Q_ref):
    model = gp.Model()
    model.setParam("OutputFlag", 1)
    #数値精度を計算速度より優先する
    model.setParam("NumericFocus", 3)
    model.setParam("ScaleFlag", 2)  # より強力なスケーリングを適用
    model.setParam("ObjScale", 0) # 目的関数のスケールを調整  
    model.setParam("BarHomogeneous", 1) 

    Q = model.addVars(cfg.K, cfg.H, lb=0)
    lam = model.addVars(cfg.N, cfg.M, lb=0)
    mu = model.addVars(cfg.N, cfg.D, lb=0)
    eps = 1e-4

    model.setObjective(
        gp.quicksum((Q[k,h] - Q_ref[k,h])**2
                    for k in range(cfg.K)
                    for h in range(cfg.H)),
        GRB.MINIMIZE
    )

    for i in range(cfg.N):
        theta = data["theta"][i]
        a = data["a_hat"][i]
        A = data["A"][i]
        b = data["b"][i]
        Sigma_inv = data["Sigma_inv"]

        w1 = sum(Q[0,h] * theta[h] for h in range(cfg.H))
        w2 = sum(Q[1,h] * theta[h] for h in range(cfg.H))

        for d in range(cfg.D):
            model.addConstr(
                2*w1*a[d]
                + 2*w2*gp.quicksum(Sigma_inv[d1][d] * a[d1] for d1 in range(cfg.D))
                + gp.quicksum(A[m,d]*lam[i,m] for m in range(cfg.M))
                - mu[i,d]
                <= eps
            )

            model.addConstr(
                2*w1*a[d]
                + 2*w2*gp.quicksum(Sigma_inv[d1][d] * a[d1] for d1 in range(cfg.D))
                + gp.quicksum(A[m,d]*lam[i,m] for m in range(cfg.M))
                - mu[i,d]
                >= -eps
            )

        for m in range(cfg.M):
            slack = sum(A[m,d]*a[d] for d in range(cfg.D)) - b[m]
            model.addConstr(lam[i,m] * slack >= -5e-3)

        for d in range(cfg.D):
            model.addConstr(mu[i,d] * a[d] <= 1e-3)
            
            
    model.addConstr(gp.quicksum(Q[k,h] for k in range(cfg.K) for h in range(cfg.H)) == 10)
    
    model.optimize()
    
    
    status = model.Status
    print("Status:", status)





    Q_est = np.array([[Q[k,h].X for h in range(cfg.H)]
                    for k in range(cfg.K)])
    return Q_est


#逆実行可能集合のチェビシェフ中心
def inverse_proposed_w_chebyshev(cfg,data):
    model = gp.Model()
    model.setParam('OutputFlag',0)
    #数値精度を計算速度より優先する
    model.setParam("NumericFocus", 3)
    
    
    #変数の定義
    Q = model.addVars(cfg.K,cfg.H,lb=0)
    lam = model.addVars(cfg.N,cfg.M,lb=0)
    mu = model.addVars(cfg.N,cfg.D,lb=0)
    tau = model.addVar(lb=0)
    eps = 1e-4
    
    #目的関数の定義
    model.setObjective(tau,GRB.MAXIMIZE)
    
    #制約式の定義
    for i in range(cfg.N):
        theta = data["theta"][i]
        a = data["a_hat"][i]
        A = data["A"][i]
        b = data["b"][i]
        Sigma_inv = data["Sigma_inv"]

        w1 = sum(Q[0,h] * theta[h] for h in range(cfg.H))
        w2 = sum(Q[1,h] * theta[h] for h in range(cfg.H))

        for d in range(cfg.D):
            model.addConstr(
                2*w1*a[d]
                + 2*w2*gp.quicksum(Sigma_inv[d1][d] * a[d1] for d1 in range(cfg.D))
                + gp.quicksum(A[m,d]*lam[i,m] for m in range(cfg.M))
                - mu[i,d]
                <= eps
            )

            model.addConstr(
                2*w1*a[d]
                + 2*w2*gp.quicksum(Sigma_inv[d1][d] * a[d1] for d1 in range(cfg.D))
                + gp.quicksum(A[m,d]*lam[i,m] for m in range(cfg.M))
                - mu[i,d]
                >= -eps
            )

        for m in range(cfg.M):
            model.addConstr(
                lam[i,m] * (sum(A[m,d]*a[d] for d in range(cfg.D)) - b[m]) >= -2e-3
            )

        for d in range(cfg.D):
            model.addConstr(mu[i,d] * a[d] <= 1e-3)
            
    #追加でチェビシェフ中心の制約式を織り込む
    for k in range(cfg.K):
        for h in range(cfg.H):
            model.addConstr(Q[k,h] - tau >= 0)
            model.addConstr(Q[k,h] + tau <= 10)
    
    #総和が1になるように
    model.addConstr(gp.quicksum(Q[k,h] for k in range(cfg.K) for h in range(cfg.H)) == 10)
    
    #modelの最適化
    model.optimize()
    
    #実際に解けているかどうかをチェックする。
    status = model.Status
    print("Status:", status)




    #Qの推定解を返す。
    Q_est = np.array([[Q[k,h].X for h in range(cfg.H)]
                    for k in range(cfg.K)])
    return Q_est
    
    
            
    