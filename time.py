import time
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from config import Config
from data_generation import generate_dataset, generate_dataset_with_noise
from inverse_problem import inverse_vi_2

def short_test(seed=0):
    np.random.seed(seed)
    # 実験するNのリスト
    NList = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    # ノイズレベルのリスト
    noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05]
    
    # 結果格納用辞書 {noise: [time_for_n1, time_for_n2, ...]}
    results_all_noise = {nl: [] for nl in noise_levels}
    num_trials = 20 

    for n in NList:
        cfg = Config(n)
        print(f"\n=============== Experiment N = {cfg.N} ================")
        
        # 各ノイズレベルごとの合計時間を初期化
        temp_totals = {nl: 0.0 for nl in noise_levels}
        
        for i in range(num_trials):
            # 1. 真のQと基礎データを生成（各試行の最初に1回）
            train_data_clean, Q_true = generate_dataset(cfg, cfg.N, cfg.seed + i)
            
            # 2. 各ノイズレベルごとに計測
            for nl in noise_levels:
                # ノイズありデータの生成（計測に含めない場合はstartの前に置く）
                train_data_noisy = generate_dataset_with_noise(cfg, cfg.N, cfg.seed + i, nl, Q_true)
                
                # --- 逆最適化の計測開始 ---
                start = time.perf_counter()
                _ = inverse_vi_2(cfg, train_data_noisy)
                end = time.perf_counter()
                # -----------------------
                
                temp_totals[nl] += (end - start)
        
        # 平均を計算してリストに格納
        for nl in noise_levels:
            avg_time = temp_totals[nl] / num_trials
            results_all_noise[nl].append(avg_time)
            print(f"  Noise {nl}: Avg Time {avg_time:.4f}s")

    return NList, results_all_noise

if __name__ == "__main__":
    # 1. フォルダの作成
    save_dir = "visualize_time"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory '{save_dir}' created.")

    # 2. 実験実行
    n_values, all_noise_times = short_test(seed=0)
    
    # 3. グラフの作成
    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'x']

    for idx, nl in enumerate(all_noise_times.keys()):
        plt.plot(n_values, all_noise_times[nl], 
                 marker=markers[idx], markersize=5, 
                 linestyle='-', linewidth=1.5, 
                 color=colors[idx], 
                 label=f'Noise Level {nl}')
    
    plt.title("Computational Scalability: Sample Size $N$ vs Execution Time", fontsize=14)
    plt.xlabel("Sample Size $N$", fontsize=12)
    plt.ylabel("Average Execution Time (seconds)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 4. 保存処理
    save_path_png = os.path.join(save_dir, "execution_time_all_noise.png")
    save_path_pdf = os.path.join(save_dir, "execution_time_all_noise.pdf")
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    
    # 5. CSV保存
    # 列名を作成 [N, noise_0.01, noise_0.02, ...]
    df_dict = {'Sample_Size_N': n_values}
    for nl in all_noise_times.keys():
        df_dict[f'Noise_{nl}'] = all_noise_times[nl]
    
    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(save_dir, "execution_time_comparison.csv"), index=False)

    print(f"\n[Completed]")
    print(f"All-in-one graph saved to: {save_path_png}")
    print(f"Data comparison CSV saved to: {os.path.join(save_dir, 'execution_time_comparison.csv')}")
    plt.show()