import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

# モジュールをimport（ノイズなしバージョンの指定構成を維持）
from config import Config
from data_generation import generate_dataset, generate_dataset_with_fixed_Q, generate_Q
from inverse_problem import inverse_proposed, inverse_proposed_w_chebyshev
from evaluation import Q_error, compute_baseline_regret, compute_regret

def run_experiment(seed=0):
    np.random.seed(seed)
    # 実験するNのリスト
    NList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    results_plot = []
    
    for n in NList:
        cfg = Config(n)
        
        print(f"=============== Experiment Configuration: N={cfg.N} ================")
        
        # 指標の平均を格納する辞書
        avg_res = {
            "mae": {"proposed": 0.0, "cheby": 0.0, "ref": 0.0},
            "rmse": {"proposed": 0.0, "cheby": 0.0, "ref": 0.0},
            "cos": {"proposed": 0.0, "cheby": 0.0, "ref": 0.0},
            "Regret": {
                "proposed": 0.0, 
                "cheby": 0.0, 
                "ref": 0.0, 
                "euclid": 0.0, 
                "maha": 0.0
            }
        }
        
        num_trials = 20
        num_refs = 3
        trial_weight = 1.0 / num_trials
        ref_weight = 1.0 / num_refs
        
        for i in range(num_trials):
            print(f"Trial {i+1}/{num_trials} (N={n})", end='\r')
            
            # [1] Generating training / validation data
            train_data, Q_true = generate_dataset(cfg, cfg.N, cfg.seed + i)
            val_data = generate_dataset_with_fixed_Q(cfg, cfg.V, cfg.seed_validation, Q_true)
            
            # [2] Solving inverse optimization with multiple reference points
            # 参照点によって結果が変わるのを防ぐため、3つの参照点で平均化
            temp_prop_mae = 0.0; temp_prop_rmse = 0.0; temp_prop_cos = 0.0; temp_prop_reg = 0.0
            temp_ref_mae = 0.0; temp_ref_rmse = 0.0; temp_ref_cos = 0.0; temp_ref_reg = 0.0
            
            for k in range(num_refs):
                Q_ref = generate_Q(cfg, k + 2)
                
                # 提案手法の推定
                Q_est_prop = inverse_proposed(cfg, train_data, Q_ref)
                
                # 指標計算
                err_prop = Q_error(Q_est_prop, Q_true)
                err_ref = Q_error(Q_ref, Q_true)
                reg_prop = compute_regret(cfg, Q_est_prop, Q_true, val_data)
                reg_ref = compute_regret(cfg, Q_ref, Q_true, val_data)
                
                # 3回分の累積
                temp_prop_mae += err_prop["mae"] * ref_weight
                temp_prop_rmse += err_prop["rmse"] * ref_weight
                temp_prop_cos += err_prop["cos"] * ref_weight
                temp_prop_reg += reg_prop * ref_weight
                
                temp_ref_mae += err_ref["mae"] * ref_weight
                temp_ref_rmse += err_ref["rmse"] * ref_weight
                temp_ref_cos += err_ref["cos"] * ref_weight
                temp_ref_reg += reg_ref * ref_weight

            # [3] Chebyshev (参照点に依存しない手法)
            Q_est_cheby = inverse_proposed_w_chebyshev(cfg, train_data)
            err_cheby = Q_error(Q_est_cheby, Q_true)
            reg_cheby = compute_regret(cfg, Q_est_cheby, Q_true, val_data)
            
            # [4] Baselines (参照点に依存しない)
            reg_euclid, reg_maha = compute_baseline_regret(cfg, Q_true, val_data)
            
            # [5] 最終的な平均への累積 (1/20)
            # MAE / RMSE / COS
            avg_res["mae"]["proposed"] += temp_prop_mae * trial_weight
            avg_res["mae"]["ref"]      += temp_ref_mae * trial_weight
            avg_res["mae"]["cheby"]    += err_cheby["mae"] * trial_weight
            
            avg_res["rmse"]["proposed"] += temp_prop_rmse * trial_weight
            avg_res["rmse"]["ref"]      += temp_ref_rmse * trial_weight
            avg_res["rmse"]["cheby"]    += err_cheby["rmse"] * trial_weight
            
            avg_res["cos"]["proposed"] += temp_prop_cos * trial_weight
            avg_res["cos"]["ref"]      += temp_ref_cos * trial_weight
            avg_res["cos"]["cheby"]    += err_cheby["cos"] * trial_weight
            
            # Regret
            avg_res["Regret"]["proposed"] += temp_prop_reg * trial_weight
            avg_res["Regret"]["ref"]      += temp_ref_reg * trial_weight
            avg_res["Regret"]["cheby"]    += reg_cheby * trial_weight
            avg_res["Regret"]["euclid"]   += reg_euclid * trial_weight
            avg_res["Regret"]["maha"]     += reg_maha * trial_weight

        print(f"\nN={n} Completed.")

        # 結果を格納
        result_plot = {
            "N": n,
            "mae": avg_res["mae"],
            "rmse": avg_res["rmse"],
            "cos": avg_res["cos"],
            "Regret": avg_res["Regret"]
        }
        results_plot.append(result_plot)
        
    return results_plot

if __name__ == "__main__":
    results = run_experiment(seed=0)
    
    save_dir = Path("results")
    save_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"exp_no_noise_{timestamp}.pkl"
    
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved to {save_path}")

    print("\n" + "="*20 + " Final Results (Averages: 20 trials x 3 refs) " + "="*20)
    for res in results:
        print(f"\nN: {res['N']} " + "-"*40)
        for metric in ["mae", "rmse", "cos", "Regret"]:
            print(f"  [{metric}]")
            for method_name, val in res[metric].items():
                print(f"    {method_name:12}: {val:.8f}")