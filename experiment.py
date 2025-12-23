import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
#ここで実験するだけ

#モジュールをimport 
from config import Config
from data_generation import generate_dataset,generate_dataset_with_fixed_Q,generate_Q
from inverse_problem import inverse_proposed,inverse_proposed_w_chebyshev
from evaluation import Q_error,compute_baseline_regret,compute_regret

def run_experiment(seed=0):
    np.random.seed(seed)
    NList = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    results = []
    results_plot = []
    
    for n in NList:
        cfg = Config(n)
        
        print("=============== Experiment Configuration ================")
        print(cfg.N)
        print("=========================================================")
        
        
        print("\n[1]  Generating training / validation data")
        
        #人工データを生成する。
        train_data,Q_true = generate_dataset(cfg,cfg.N,cfg.seed)
        val_data = generate_dataset_with_fixed_Q(cfg,cfg.V,cfg.seed_validation,Q_true)
        
        Q_ref = generate_Q(cfg,cfg.seed_validation)

        

        
        print("\n[2] Solving inverse optimization (Proposed method)")
        #逆最適化を使って推定解を導出する。
        Q_est = inverse_proposed(cfg,train_data,Q_ref)
        Q_est_cheby = inverse_proposed_w_chebyshev(cfg,train_data)
        
        print("\n[3] Q estimation error")
        #Qの誤差を表示する。
        Q_gosa = Q_error(Q_est,Q_true)
        Q_gosa_cheby = Q_error(Q_est_cheby,Q_true)
        print(Q_gosa)
        print(Q_gosa_cheby)
        
        print("\n[4] Regret evaluation (Proposed method))")
        
        #既存手法と提案手法のRegretを計算する。
        regret_proposed = compute_regret(cfg,Q_est,Q_true,val_data)
        regret_ref_proposed = compute_regret(cfg,Q_ref,Q_true,val_data)
        regret_cheby_proposed = compute_regret(cfg,Q_est_cheby,Q_true,val_data)
        print(f"Proposed method regret = {regret_proposed:.6f}")
        print(f"Proposed method regret(ref) = {regret_ref_proposed:.6f}")
        print(f"Proposed method with chebyshev regret = {regret_cheby_proposed:.6f}")
        
        print("\n[5] Regret evaluation (baseline method)")
        regret_euclid,regret_maha = compute_baseline_regret(cfg,Q_true,val_data)
        print(f"Euclidean baseline regret    = {regret_euclid:.6f}")
        print(f"Mahalanobis baseline regret  = {regret_maha:.6f}")
        
        
        result = {
            "N": n,
            "Q_error": Q_gosa,
            "Q_error_cheby": Q_gosa_cheby,
            "regret_proposed": regret_proposed,
            "regret_ref": regret_ref_proposed,
            "regret_cheby_proposed":regret_cheby_proposed,
            "regret_euclid": regret_euclid,
            "regret_maha": regret_maha
        }
        
        #可視化用のデータ形式も作る。
        result_plot = {
            "N" : n,
            
            "Q_error": {
                "proposed": Q_gosa,
                "cheby": Q_gosa_cheby,
            },
            "regret": {
                "proposed": regret_proposed,
                "ref": regret_ref_proposed,
                "cheby": regret_cheby_proposed,
                "euclid": regret_euclid,
                "maha": regret_maha,
            }
        }
        
        #実験結果を格納する。
        results.append(result)
        results_plot.append(result_plot)


    return results,results_plot
    
    
if __name__ == "__main__":
    results,results_plot = run_experiment(seed=0)
    
    #result_plotを保存する場所を提供する。
    save_dir = Path("results")
    save_dir.mkdir(exist_ok=True)
    #タイムスタンプの作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"exp_{timestamp}.pkl"
    
    
    with open(save_path, "wb") as f:
        pickle.dump(results_plot, f)

    print(f"Saved to {save_path}")

    print("\n===== Final Results =====")
    for i in range(len(results)):
        N = results[i]["N"]
        print(f"N:{N}-----------------")
        for k, v in results[i].items():
            print(f"{k}: {v:.6f}")
    

    
    
