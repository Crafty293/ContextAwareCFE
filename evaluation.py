import numpy as np
from forward_problem import solve_forward
from baseline import solve_euclid,solve_maha

#真のQと推定したQの比較をする。
def Q_error(Q_est,Q_true):
    rmse = Q_rmse(Q_est,Q_true)
    mae = Q_mae(Q_est,Q_true)
    cos = Q_cos(Q_est,Q_true)
    
    return {"rmse":rmse,"mae":mae,"cos":cos}


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

#提案手法のregretを計算する(検証用)
def compute_regret_val(cfg,Q_est,Q_true,val_data):
    regret = 0.0
    
    for i in range(cfg.N):
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
        
    return regret / cfg.N


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

        


#Qの評価についてもう少し項目を増加する

#推定した行列のcos類似を計算するQの形はnp.array
def Q_cos(Q_est, Q_true):
    # 行列(2x15)を1次元(30次元)のベクトルに平坦化する
    est_flat = Q_est.flatten()
    true_flat = Q_true.flatten()
    
    # ベクトルのノルムを計算
    norm_est = np.linalg.norm(est_flat)
    norm_true = np.linalg.norm(true_flat)
    
    # 内積をノルムの積で割る（0除算対策つき）
    cos_sim = np.dot(est_flat, true_flat) / (norm_est * norm_true + 1e-8)
    
    return cos_sim # これでスカラー（1つの数値）が返ります

#推定した行列の要素ごとの誤差を計算する。
def Q_mae(Q_est,Q_true):
    mae = np.mean(np.abs(Q_est - Q_true))
    return mae

#推定した行列のrmseを計算する。
def Q_rmse(Q_est,Q_true):
    rmse = np.sqrt(np.mean((Q_est - Q_true)**2))
    return rmse

