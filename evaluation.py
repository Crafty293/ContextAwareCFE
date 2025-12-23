import numpy as np
from forward_problem import solve_forward
from baseline import solve_euclid,solve_maha

#真のQと推定したQの比較をする。
def Q_error(Q_est,Q_true):
    return np.linalg.norm(Q_est - Q_true)

#実際のコストを計算する。
def compute_cost(cfg,a,Q,theta,Sigma_inv):
    a = np.array(a)
    
    w1 = sum(Q[0,h] * theta[h] for h in range(cfg.H))
    w2 = sum(Q[1,h] * theta[h] for h in range(cfg.H))
    
    cost = 0.0
    cost += w1 * sum(a[d] * a[d] for d in range(cfg.D))
    cost += w2 * sum(Sigma_inv[d1][d2] * a[d1] * a[d2] for d1 in range(cfg.D) for d2 in range(cfg.D))
    
    return cost

#提案手法のregretを計算する
def compute_regret(cfg,Q_est,Q_true,val_data):
    regret = 0.0
    
    for i in range(cfg.V):
        theta = val_data["theta"][i]
        A = val_data["A"][i]
        b = val_data["b"][i]
        Sigma_inv = val_data["Sigma_inv"]
        
        #推定Qで出したアクションを計算する。
        a_est = solve_forward(cfg,Q_est,theta,A,b,Sigma_inv)
        
        #真の最適行動(すでにdata_generationで生成)
        a_true = val_data["a_hat"][i]
        
        #真のQでの評価
        cost_est = compute_cost(cfg,a_est,Q_true,theta,Sigma_inv)
        cost_true = compute_cost(cfg,a_true,Q_true,theta,Sigma_inv)
        
        regret += cost_est - cost_true
        
    return regret / cfg.V


#baselineのregretを推定する。
def compute_baseline_regret(cfg,Q_true,val_data):
    regret_euclid = 0.0
    regret_maha = 0.0
    
    for i in range(cfg.V):
        A = val_data["A"][i]
        b = val_data["b"][i]
        theta = val_data["theta"][i]
        Sigma_inv = val_data["Sigma_inv"]
        
        #既存手法で出したアクションを計算する。
        a_euclid = solve_euclid(cfg,A,b)
        a_maha = solve_maha(cfg,A,b,Sigma_inv)
        
        #真の最適アクション(すでにdata_generationで生成)
        a_true = val_data["a_hat"][i]
        
        #真のコスト関数での評価
        cost_true = compute_cost(cfg,a_true,Q_true,theta,Sigma_inv)
        cost_euclid = compute_cost(cfg,a_euclid,Q_true,theta,Sigma_inv)
        cost_maha = compute_cost(cfg,a_maha,Q_true,theta,Sigma_inv)
        
        regret_euclid += cost_euclid - cost_true
        regret_maha += cost_maha - cost_true
        
    return regret_euclid / cfg.V ,regret_maha / cfg.V

        
