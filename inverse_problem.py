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
                + 2*w2*gp.quicksum(Sigma_inv[d][d1] * a[d1] for d1 in range(cfg.D))
                + gp.quicksum(A[m,d]*lam[i,m] for m in range(cfg.M))
                - mu[i,d]
                <= eps
            )

            model.addConstr(
                2*w1*a[d]
                + 2*w2*gp.quicksum(Sigma_inv[d][d1] * a[d1] for d1 in range(cfg.D))
                + gp.quicksum(A[m,d]*lam[i,m] for m in range(cfg.M))
                - mu[i,d]
                >= -eps
            )

        for m in range(cfg.M):
            slack = sum(A[m,d]*a[d] for d in range(cfg.D)) - b[m]
            model.addConstr(lam[i,m] * slack >= -5e-3)

        for d in range(cfg.D):
            model.addConstr(mu[i,d] * a[d] <= 1e-3)
            
            
    model.addConstr(gp.quicksum(Q[k,h] for k in range(cfg.K) for h in range(cfg.H)) == 1)
    
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
                + 2*w2*gp.quicksum(Sigma_inv[d][d1] * a[d1] for d1 in range(cfg.D))
                + gp.quicksum(A[m,d]*lam[i,m] for m in range(cfg.M))
                - mu[i,d]
                <= eps
            )

            model.addConstr(
                2*w1*a[d]
                + 2*w2*gp.quicksum(Sigma_inv[d][d1] * a[d1] for d1 in range(cfg.D))
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
            model.addConstr(Q[k,h] + tau <= 1)
    
    #総和が1になるように
    model.addConstr(gp.quicksum(Q[k,h] for k in range(cfg.K) for h in range(cfg.H)) == 1)
    
    #modelの最適化
    model.optimize()
    
    #実際に解けているかどうかをチェックする。
    status = model.Status
    print("Status:", status)




    #Qの推定解を返す。
    Q_est = np.array([[Q[k,h].X for h in range(cfg.H)]
                    for k in range(cfg.K)])
    return Q_est
    
    
            


#Geminiが作ったやつ優秀!!!!
def inverse_vi_2(cfg, data):
    b_l = data["b"]
    a_hat_all = data["a_hat"] # 観測値

    model = gp.Model()
    model.setParam("OutputFlag", 0)
    model.setParam("NumericFocus", 3)
    model.setParam("ScaleFlag", 2)
    
    # 変数の定義
    Q = model.addVars(cfg.K, cfg.H, lb=0, name="Q")
    y = model.addVars(cfg.N, cfg.M, lb=0, name="y")
    
    # 目的関数のパーツを格納するリスト
    obj_terms = []
    
    # 制約式の定義と目的関数の構築
    for i in range(cfg.N):
        theta = data["theta"][i]
        a_i = a_hat_all[i] # このデータセットにおける a_hat
        A = data["A"][i]
        Sigma_inv = data["Sigma_inv"]

        # 勾配の重み（Qに依存する変数）
        w1 = gp.quicksum(Q[0, h] * theta[h] for h in range(cfg.H))
        w2 = gp.quicksum(Q[1, h] * theta[h] for h in range(cfg.H))
        
        # --- 目的関数のための「定数項（a_hatの項）」の計算 ---
        # ∇g1(a_i)^T * a_i = 2 * a_i^T * a_i
        grad_g1_dot_a = 2 * sum(a_i[d] * a_i[d] for d in range(cfg.D))
        
        # ∇g2(a_i)^T * a_i = 2 * (Sigma_inv * a_i)^T * a_i
        # まずベクトル Σ^-1 * a_i を計算
        sigma_inv_a = [sum(Sigma_inv[d][d1] * a_i[d1] for d1 in range(cfg.D)) for d in range(cfg.D)]
        grad_g2_dot_a = 2 * sum(sigma_inv_a[d] * a_i[d] for d in range(cfg.D))
        
        # このデータ i における損失関数の構成要素を追加
        # l_vi_i = b_i^T * y_i + w1 * (grad_g1_dot_a) + w2 * (grad_g2_dot_a)
        obj_terms.append(gp.quicksum(b_l[i][m] * y[i, m] for m in range(cfg.M)))
        obj_terms.append(w1 * grad_g1_dot_a)
        obj_terms.append(w2 * grad_g2_dot_a)
        
        # --- 制約式の定義 (A^T * y >= -grad) ---
        for d in range(cfg.D):
            # 勾配の d 成分
            grad_val_d = (2 * w1 * a_i[d] + 
                          2 * w2 * gp.quicksum(Sigma_inv[d][d1] * a_i[d1] for d1 in range(cfg.D)))
            model.addConstr(
                gp.quicksum(A[m][d] * y[i, m] for m in range(cfg.M)) >= -grad_val_d,
                name=f"dual_cons_{i}_{d}"
            )
    
    # 目的関数のセット（全データの平均損失を最小化）
    model.setObjective(gp.quicksum(obj_terms) / cfg.N, GRB.MINIMIZE)
    
    # Qの正規化条件
    model.addConstr(gp.quicksum(Q[k, h] for k in range(cfg.K) for h in range(cfg.H)) == 1)
    
    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        print("Optimal solution found.")
        Q_est = np.array([[Q[k, h].X for h in range(cfg.H)] for k in range(cfg.K)])
        return Q_est
    else:
        print(f"Optimization ended with status {model.Status}")
        # Status 4 (INF_OR_UNBD) の場合は、ここをチェック
        return None