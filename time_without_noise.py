import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path

# モジュールをimport
from config import Config
from data_generation import generate_dataset, generate_Q
from inverse_problem import inverse_proposed, inverse_proposed_w_chebyshev

def run_time_experiment(seed=0):
    np.random.seed(seed)
    # 実験するNのリスト
    NList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    results_time = []
    
    num_trials = 20
    
    for n in NList:
        cfg = Config(n)
        print(f"=============== Measuring Time: N={cfg.N} ================")
        
        total_time_proposed = 0.0
        total_time_cheby = 0.0
        
        for i in range(num_trials):
            # データの準備（計測には含めない）
            train_data, _ = generate_dataset(cfg, cfg.N, cfg.seed + i)
            Q_ref = generate_Q(cfg, seed=i+100) # 参照点
            
            # [1] Proposed の時間計測
            start_p = time.perf_counter()
            _ = inverse_proposed(cfg, train_data, Q_ref)
            end_p = time.perf_counter()
            total_time_proposed += (end_p - start_p)
            
            # [2] Chebyshev の時間計測
            start_c = time.perf_counter()
            _ = inverse_proposed_w_chebyshev(cfg, train_data)
            end_c = time.perf_counter()
            total_time_cheby += (end_c - start_c)
            
            print(f"Trial {i+1}/{num_trials} (N={n}) done.", end='\r')

        # 平均時間を算出
        avg_p = total_time_proposed / num_trials
        avg_c = total_time_cheby / num_trials
        
        results_time.append({
            "N": n,
            "Proposed": avg_p,
            "Chebyshev": avg_c
        })
        print(f"\nN={n}: Proposed={avg_p:.4f}s, Cheby={avg_c:.4f}s")
        
    return results_time

if __name__ == "__main__":
    # 1. 保存ディレクトリの設定
    save_dir = Path("result_time_without_noise")
    save_dir.mkdir(exist_ok=True)
    
    # 2. 実験実行
    results = run_time_experiment(seed=0)
    
    # 3. データの整理
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV保存
    csv_path = save_dir / f"time_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # 4. 可視化
    plt.figure(figsize=(10, 6))
    plt.plot(df["N"], df["Proposed"], marker='o', label="Proposed (with Ref)", linewidth=1.5)
    plt.plot(df["N"], df["Chebyshev"], marker='s', label="Chebyshev", linewidth=1.5)
    
    plt.title("Execution Time Comparison: Proposed vs Chebyshev", fontsize=14)
    plt.xlabel("Sample Size $N$", fontsize=12)
    plt.ylabel("Execution Time (seconds)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # グラフ保存
    plot_path = save_dir / f"time_comparison_plot_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"\n[Completed]")
    print(f"Data saved to: {csv_path}")
    print(f"Plot saved to: {plot_path}")
    plt.show()