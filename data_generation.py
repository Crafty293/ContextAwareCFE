from config import Config
from forward_problem import solve_forward,solve_forward_w_noise
import numpy as np


#真のQを生成する。(K*H)
def generate_Q(cfg,seed=None):
    rng = np.random.default_rng(seed)
    Q = rng.uniform(0,8,size=(cfg.K, cfg.H))
    Q /= Q.sum()
    return Q

#コンテキストを生成する((N+V)*H) → 結局こっち使うことになりそう。
def generate_theta(cfg,num,seed=None):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0,4,size=(num,cfg.H))
    theta[np.abs(theta) < 1e-4] = 0
    return theta

#コンテキストの生成(改良版)
def generate_theta_2(cfg, num, sigma2, seed=None):
    rng = np.random.default_rng(seed)
    # 各次元ごとに上限を変える
    sigma2_theta = sigma2[:cfg.H]
    upper = np.sqrt(12 * sigma2_theta)   # shape (H,)
    theta = rng.uniform(
        low=0.0,
        high=upper,
        size=(num, cfg.H)
    )
    return theta

#分散共分散行列として対角行列を禁止されたので頑張る。LL^{T}を作ってその逆行列を渡す。(実際に使うのは分散逆分散行列でなくその逆行列だから)
def generate_Sigma_inv(cfg, seed=None):
    rng = np.random.default_rng(seed)

    M = rng.normal(0, 1, size=(cfg.D, cfg.D))
    M[np.abs(M) < 1e-4] = 0
    Sigma_inv = M.T @ M + 1.0 * np.eye(cfg.D)
    # 対角成分以外（非対角成分）で、絶対値が小さいものを消す
    for r in range(cfg.D):
        for c in range(cfg.D):
            if r != c: # 対角線以外
                if abs(Sigma_inv[r, c]) < 1e-3:
                    Sigma_inv[r, c] = 0
    return Sigma_inv



#制約式を生成する。
def generate_constraints(cfg, num, seed=None):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-2, -1, size=(num, cfg.M, cfg.D))
    
    b = rng.uniform(-20, -11, size=(num, cfg.M))
    return A, b

#xの特徴量の各分散を計算(D個生成)
def generate_sigma2(cfg, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(1,10,size=cfg.D)



#データまとめて生成する
def generate_dataset(cfg, num, seed):
    Q_true = generate_Q(cfg,seed)
    Sigma_inv = generate_Sigma_inv(cfg,seed)
    theta = generate_theta(cfg,num,seed)
    A, b = generate_constraints(cfg, num,seed)

    a_hat = []
    for i in range(num):
        a = solve_forward(cfg, Q_true, theta[i], A[i], b[i],Sigma_inv)
        a = np.array(a)
        a_hat.append(a)

    return {
        "theta": theta,
        "A": A,
        "b": b,
        "a_hat": np.array(a_hat),
        "Sigma_inv": Sigma_inv
    },Q_true

#特定のQに対してデータを作成する(検証用データの作成にはこっちを使う。)    
def generate_dataset_with_fixed_Q(cfg, num, seed, Q_true):
    Sigma_inv = generate_Sigma_inv(cfg,seed)
    theta= generate_theta(cfg,num,seed)
    A, b = generate_constraints(cfg, num,seed)

    a_hat = []
    for i in range(num):
        a = solve_forward(cfg, Q_true, theta[i], A[i], b[i], Sigma_inv)
        a_hat.append(a)

    return {
        "theta": theta,
        "A": A,
        "b": b,
        "a_hat": np.array(a_hat),
        "Sigma_inv": Sigma_inv
    }

def generate_noisy_Q(true_Q,sigma):
    #真のQに対してノイズを加える。
    noisy_Q = true_Q + np.random.normal(0,sigma,size=true_Q.shape)
    
    #非負制約と正規化を再適用
    noisy_Q = np.maximum(noisy_Q,0)
    noisy_Q /= np.sum(noisy_Q)
    
    return noisy_Q

#ランダムノイズを作る。
def make_noise_for_instance(sigma,size,i):
    rng = np.random.default_rng(999+i)
    return rng.normal(0,sigma,size)

#ノイズありの時にデータをまとめて生成する。
def generate_dataset_with_noise(cfg,num,seed,sigma,Q_true):
    Sigma_inv = generate_Sigma_inv(cfg,seed)
    theta = generate_theta(cfg,num,seed)
    A, b = generate_constraints(cfg, num,seed)

    a_hat = []
    for i in range(num):
        noise = make_noise_for_instance(sigma,2,i)
        a = solve_forward_w_noise(cfg, Q_true, theta[i], A[i], b[i],Sigma_inv,noise[0],noise[1])
        a = np.array(a)
        a_hat.append(a)

    return {
        "theta": theta,
        "A": A,
        "b": b,
        "a_hat": np.array(a_hat),
        "Sigma_inv": Sigma_inv
    }
