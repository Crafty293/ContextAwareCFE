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
        
        #結果に載せる指標の空箱を準備する
        Q_gosa_ref_ave = {"rmse":0.0,"mae":0.0,"cos":0.0}
        Q_gosa_vi_001_ave = {"rmse":0.0,"mae":0.0,"cos":0.0}
        Q_gosa_vi_002_ave = {"rmse":0.0,"mae":0.0,"cos":0.0}
        Q_gosa_vi_003_ave = {"rmse":0.0,"mae":0.0,"cos":0.0}
        Q_gosa_vi_004_ave = {"rmse":0.0,"mae":0.0,"cos":0.0}
        Q_gosa_vi_005_ave = {"rmse":0.0,"mae":0.0,"cos":0.0}
        regret_ref_ave = 0.0
        regret_vi_001_ave = 0.0
        regret_vi_002_ave = 0.0
        regret_vi_003_ave = 0.0
        regret_vi_004_ave = 0.0
        regret_vi_005_ave = 0.0
        
        
        
        #複数のデータ使って実験してみる。
        for i in range(20):
            
            print("\n[1]  Generating training / validation data")
            
            print("ノイズなしデータ作成開始")
            #人工データを生成する。(今回検証用実験なためtrain_dataのみ)
            train_data,Q_true = generate_dataset(cfg,cfg.N,cfg.seed+i)
            val_data = generate_dataset_with_fixed_Q(cfg,cfg.V,cfg.seed_validation,Q_true)
            
            print("ノイズなしデータ生成完了")
            
            print("ノイズありデータ作成開始")
            #noiseありのデータを作成する。
            train_data_001 = generate_dataset_with_noise(cfg,cfg.N,cfg.seed+i,0.01,Q_true)
            train_data_002 = generate_dataset_with_noise(cfg,cfg.N,cfg.seed+i,0.02,Q_true)
            train_data_003 = generate_dataset_with_noise(cfg,cfg.N,cfg.seed+i,0.03,Q_true)
            train_data_004 = generate_dataset_with_noise(cfg,cfg.N,cfg.seed+i,0.04,Q_true)
            train_data_005 = generate_dataset_with_noise(cfg,cfg.N,cfg.seed+i,0.05,Q_true)
            
            print("ノイズありデータ作成完了")
            
            print("ナイーブ手法検証開始")
            #参照点での評価指標を計算する。
            regret_ref = 0.0
            Q_ref_mae = 0.0
            Q_ref_rmse = 0.0
            Q_ref_cos = 0.0
            for j in range(3):
                Q_ref = generate_Q(cfg,cfg.seed_validation-j)
                regret_ref+=compute_regret(cfg,Q_ref,Q_true,val_data)/3
                Q_ref_error = Q_error(Q_ref,Q_true)
                Q_ref_mae += Q_ref_error["mae"]
                Q_ref_rmse += Q_ref_error["rmse"]
                Q_ref_cos += Q_ref_error["cos"]
            
            #平均を取る
            Q_ref_mae = Q_ref_mae/3
            Q_ref_rmse = Q_ref_rmse/3
            Q_ref_cos = Q_ref_cos/3
            
            print("ナイーブ手法完了")
            
            
            
            
            print("提案手法開始")
            print("\n[2] Solving inverse optimization (Proposed method)")
            #逆最適化を使って推定解を導出する。
            Q_est = inverse_vi_2(cfg,train_data)
            Q_est_001 = inverse_vi_2(cfg,train_data_001)
            Q_est_002 = inverse_vi_2(cfg,train_data_002)
            Q_est_003 = inverse_vi_2(cfg,train_data_003)
            Q_est_004 = inverse_vi_2(cfg,train_data_004)
            Q_est_005 = inverse_vi_2(cfg,train_data_005)

            
            print("提案手法推定完了")
            print("\n[3] Q estimation error")
            #Qの誤差を表示する。
            print("誤差測定中")
            Q_gosa_vi_001 = Q_error(Q_est_001,Q_true)
            Q_gosa_vi_002 = Q_error(Q_est_002,Q_true)
            Q_gosa_vi_003 = Q_error(Q_est_003,Q_true)
            Q_gosa_vi_004 = Q_error(Q_est_004,Q_true)
            Q_gosa_vi_005 = Q_error(Q_est_005,Q_true)
            print("誤差測定終了")
            
            
            #推定したQのrgretを計算する。
            print("regret計算中")
            regret_vi_001 = compute_regret(cfg,Q_est_001,Q_true,val_data)
            regret_vi_002 = compute_regret(cfg,Q_est_002,Q_true,val_data)
            regret_vi_003 = compute_regret(cfg,Q_est_003,Q_true,val_data)
            regret_vi_004 = compute_regret(cfg,Q_est_004,Q_true,val_data)
            regret_vi_005 = compute_regret(cfg,Q_est_005,Q_true,val_data)
            print("regret計算終了")
        
            #結果を格納する。
            Q_gosa_ref_ave["mae"] += Q_ref_mae/20
            Q_gosa_ref_ave["rmse"] += Q_ref_rmse/20
            Q_gosa_ref_ave["cos"] += Q_ref_cos/20
            regret_ref_ave += regret_ref/20
            
            Q_gosa_vi_001_ave["cos"] += Q_gosa_vi_001["cos"]/20
            Q_gosa_vi_001_ave["rmse"] += Q_gosa_vi_001["rmse"]/20
            Q_gosa_vi_001_ave["mae"] += Q_gosa_vi_001["mae"]/20
            Q_gosa_vi_002_ave["cos"] += Q_gosa_vi_002["cos"]/20
            Q_gosa_vi_002_ave["rmse"] += Q_gosa_vi_002["rmse"]/20
            Q_gosa_vi_002_ave["mae"] += Q_gosa_vi_002["mae"]/20
            Q_gosa_vi_003_ave["cos"] += Q_gosa_vi_003["cos"]/20
            Q_gosa_vi_003_ave["rmse"] += Q_gosa_vi_003["rmse"]/20
            Q_gosa_vi_003_ave["mae"] += Q_gosa_vi_003["mae"]/20
            Q_gosa_vi_004_ave["cos"] += Q_gosa_vi_004["cos"]/20
            Q_gosa_vi_004_ave["rmse"] += Q_gosa_vi_004["rmse"]/20
            Q_gosa_vi_004_ave["mae"] += Q_gosa_vi_004["mae"]/20
            Q_gosa_vi_005_ave["cos"] += Q_gosa_vi_005["cos"]/20
            Q_gosa_vi_005_ave["rmse"] += Q_gosa_vi_005["rmse"]/20
            Q_gosa_vi_005_ave["mae"] += Q_gosa_vi_005["mae"]/20
            regret_vi_001_ave += regret_vi_001/20
            regret_vi_002_ave += regret_vi_002/20
            regret_vi_003_ave += regret_vi_003/20
            regret_vi_004_ave += regret_vi_004/20
            regret_vi_005_ave += regret_vi_005/20
            
        


        
        result_plot = {
            "N": n,
            "mae": {
                "ref":Q_gosa_ref_ave["mae"],
                "noise001":Q_gosa_vi_001_ave["mae"],
                "noise002":Q_gosa_vi_002_ave["mae"],
                "noise003":Q_gosa_vi_003_ave["mae"],
                "noise004":Q_gosa_vi_004_ave["mae"],
                "noise005":Q_gosa_vi_005_ave["mae"],
            },
            "rmse":{
                "ref":Q_gosa_ref_ave["rmse"],
                "noise001":Q_gosa_vi_001_ave["rmse"],
                "noise002":Q_gosa_vi_002_ave["rmse"],
                "noise003":Q_gosa_vi_003_ave["rmse"],
                "noise004":Q_gosa_vi_004_ave["rmse"],
                "noise005":Q_gosa_vi_005_ave["rmse"],               
            },
            "cos":{
                "ref":Q_gosa_ref_ave["cos"],
                "noise001":Q_gosa_vi_001_ave["cos"],
                "noise002":Q_gosa_vi_002_ave["cos"],
                "noise003":Q_gosa_vi_003_ave["cos"],
                "noise004":Q_gosa_vi_004_ave["cos"],
                "noise005":Q_gosa_vi_005_ave["cos"],     
            },
            "Regret": {
                "ref":regret_ref_ave,
                "noise001": regret_vi_001_ave,
                "noise002": regret_vi_002_ave,
                "noise003": regret_vi_003_ave,
                "noise004": regret_vi_004_ave,
                "noise005": regret_vi_005_ave,
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