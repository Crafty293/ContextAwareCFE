#こっちはノイズありの実験場所にする。

import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

#モジュールをimport 
from config import Config
from data_generation import generate_dataset,generate_dataset_with_fixed_Q,generate_noisy_Q,generate_Q,generate_dataset_with_noise
from inverse_problem import inverse_vi_2
from evaluation import Q_error,compute_regret_val,compute_regret

def short_test(seed=0):
    np.random.seed(seed)
    NList = [1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
    results_plot = []
    
    for n in  NList:
        cfg = Config(n)

        print("=============== Experiment Configuration ================")
        print(cfg.N)
        print("=========================================================")
        
        
        print("\n[1]  Generating training / validation data")
        
        #人工データを生成する。(今回検証用実験なためtrain_dataのみ)
        train_data,Q_true = generate_dataset(cfg,cfg.N,cfg.seed)
        val_data = generate_dataset_with_fixed_Q(cfg,cfg.V,cfg.seed_validation,Q_true)
        
        #noiseありのデータを作成する。
        train_data_001 = generate_dataset_with_noise(cfg,cfg.N,cfg.seed,0.001,Q_true)
        train_data_002 = generate_dataset_with_noise(cfg,cfg.N,cfg.seed,0.002,Q_true)
        train_data_003 = generate_dataset_with_noise(cfg,cfg.N,cfg.seed,0.003,Q_true)
        train_data_004 = generate_dataset_with_noise(cfg,cfg.N,cfg.seed,0.004,Q_true)
        train_data_005 = generate_dataset_with_noise(cfg,cfg.N,cfg.seed,0.005,Q_true)
        
        
        #w(theta)の平均を計算して大まかなノイズを決めよう。(検証)
        wtheta_1 = 0.0
        wtheta_2 = 0.0
        for i in range(cfg.N):
            theta_i = train_data['theta'][i]
            wtheta_1+= sum(Q_true[0,h] * theta_i[h] for h in range(cfg.H))
            wtheta_2+= sum(Q_true[1,h] * theta_i[h] for h in range(cfg.H))
        
        wtheta_1 = wtheta_1 / cfg.N
        wtheta_2 = wtheta_2 / cfg.N
        
        print("重みの平均---------")
        print(f"重み1個目 {wtheta_1}")
        print(f"重み2個目 {wtheta_2}")
            
            
        #参照点での評価指標を計算する。
        regret_ref = 0.0
        Q_ref_mae = 0.0
        Q_ref_rmse = 0.0
        Q_ref_cos = 0.0
        for i in range(100):
            Q_ref = generate_Q(cfg,cfg.seed_validation-i)
            regret_ref+=compute_regret(cfg,Q_ref,Q_true,val_data)/100
            Q_ref_error = Q_error(Q_ref,Q_true)
            Q_ref_mae += Q_ref_error["mae"]
            Q_ref_rmse += Q_ref_error["rmse"]
            Q_ref_cos += Q_ref_error["cos"]
        
        #平均を取る
        Q_ref_mae = Q_ref_mae/100
        Q_ref_rmse = Q_ref_rmse/100
        Q_ref_cos = Q_ref_cos/100
        
        
        
        
        
        print("\n[2] Solving inverse optimization (Proposed method)")
        #逆最適化を使って推定解を導出する。
        Q_est = inverse_vi_2(cfg,train_data)
        Q_est_001 = inverse_vi_2(cfg,train_data_001)
        Q_est_002 = inverse_vi_2(cfg,train_data_002)
        Q_est_003 = inverse_vi_2(cfg,train_data_003)
        Q_est_004 = inverse_vi_2(cfg,train_data_004)
        Q_est_005 = inverse_vi_2(cfg,train_data_005)

        
        
        print("\n[3] Q estimation error")
        #Qの誤差を表示する。
        Q_gosa_vi_001 = Q_error(Q_est_001,Q_true)
        Q_gosa_vi_002 = Q_error(Q_est_002,Q_true)
        Q_gosa_vi_003 = Q_error(Q_est_003,Q_true)
        Q_gosa_vi_004 = Q_error(Q_est_004,Q_true)
        Q_gosa_vi_005 = Q_error(Q_est_005,Q_true)

        
        
        #推定したQのrgretを計算する。
        regret_vi_001 = compute_regret(cfg,Q_est_001,Q_true,val_data)
        regret_vi_002 = compute_regret(cfg,Q_est_002,Q_true,val_data)
        regret_vi_003 = compute_regret(cfg,Q_est_003,Q_true,val_data)
        regret_vi_004 = compute_regret(cfg,Q_est_004,Q_true,val_data)
        regret_vi_005 = compute_regret(cfg,Q_est_005,Q_true,val_data)

        
        result_plot = {
            "N": n,
            "mae": {
                "ref":Q_ref_mae,
                "noise001":Q_gosa_vi_001["mae"],
                "noise002":Q_gosa_vi_002["mae"],
                "noise003":Q_gosa_vi_003["mae"],
                "noise004":Q_gosa_vi_004["mae"],
                "noise005":Q_gosa_vi_005["mae"],
            },
            "rmse":{
                "ref":Q_ref_rmse,
                "noise001":Q_gosa_vi_001["rmse"],
                "noise002":Q_gosa_vi_002["rmse"],
                "noise003":Q_gosa_vi_003["rmse"],
                "noise004":Q_gosa_vi_004["rmse"],
                "noise005":Q_gosa_vi_005["rmse"],               
            },
            "cos":{
                "ref":Q_ref_cos,
                "noise001":Q_gosa_vi_001["cos"],
                "noise002":Q_gosa_vi_002["cos"],
                "noise003":Q_gosa_vi_003["cos"],
                "noise004":Q_gosa_vi_004["cos"],
                "noise005":Q_gosa_vi_005["cos"],     
            },
            "Regret": {
                "ref":regret_ref,
                "noise001": regret_vi_001,
                "noise002": regret_vi_002,
                "noise003": regret_vi_003,
                "noise004": regret_vi_004,
                "noise005": regret_vi_005,
            }
        }
        
        
        results_plot.append(result_plot)
        
    return results_plot

if __name__== "__main__":
    results = short_test()
    
    #結果を保存するためのディレクトリもつくる。
    save_dir = Path('results_w_noise')
    save_dir.mkdir(exist_ok=True)

    #タイムスタンプの作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"exp_{timestamp}.pkl"
    
    print(f"Saved to {save_path}")
    
    with open(save_path,"wb") as f:
        pickle.dump(results,f)

    print("\n===== Final Results =====")
    for result in results:
        # Nは辞書ではないので個別に取り出す
        print(f"\nN: {result['N']} " + "-"*30)
        
        # Q_error と Regret のループ
        for metric in ["Q_error", "Regret"]:
            print(f"  [{metric}]")
            # ここは辞書なので .items() でループ回せる
            for label, value in result[metric].items():
                print(f"    {label:15}: {value:.6f}")