import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_results(pkl_path):
    with open(pkl_path, "rb") as f:
        results_plot = pickle.load(f)
    return results_plot

def plot_noise_comparison(results_plot, save_dir, metric_key, ylabel, title, filename, use_log=False, y_limit=None):
    """
    metric_key: "mae", "rmse", "cos", "Regret"
    """
    # Nのリスト (x軸)
    N_list = [r["N"] for r in results_plot]
    
    # データの各ラベル (ref, noise001, noise002...) を取得
    # keyが存在しない場合を考慮して安全に取得
    available_labels = list(results_plot[0][metric_key].keys())

    plt.figure(figsize=(10, 6))
    # 見分けやすいようにマーカーを多めに設定
    markers = ["o", "s", "^", "D", "v", "p", "*", "x"]
    
    for i, label in enumerate(available_labels):
        # 各Nにおける値をリスト化
        values = [r[metric_key][label] for r in results_plot]
        
        plt.plot(N_list, values, 
                 marker=markers[i % len(markers)], 
                 label=label, 
                 markersize=6, 
                 linewidth=1.5)

    plt.xlabel("Number of training samples (N)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    
    if use_log:
        plt.yscale("log")
    
    # 指標に応じた自動範囲設定（y_limitが指定されていない場合）
    if y_limit:
        plt.ylim(y_limit[0], y_limit[1])
    else:
        if metric_key == "cos":
            # コサイン類似度は 1.0 に近いほど良いため、上の方を表示
            plt.ylim(min(0.0, plt.ylim()[0]), 1.05)
        elif metric_key == "Regret" and not use_log:
            plt.ylim(0, None)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = save_dir / filename
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Successfully saved: {save_path}")

if __name__ == "__main__":
    result_dir = Path("results_w_noise")
    visualize_dir = Path("visualize_noise")
    visualize_dir.mkdir(exist_ok=True)

    # 最新のファイルを取得
    pkl_files = sorted(result_dir.glob("exp_*.pkl"))
    if not pkl_files:
        print(f"No result files found in {result_dir}")
    else:
        pkl_path = pkl_files[-1]
        print(f"Processing: {pkl_path}")
        results_data = load_results(pkl_path)

        # --- 1. MAE (Mean Absolute Error) ---
        plot_noise_comparison(
            results_data, visualize_dir, "mae", 
            ylabel="MAE (Lower is better)", 
            title="Q Estimation: Mean Absolute Error", 
            filename="noise_MAE.png"
        )

        # --- 2. RMSE (Root Mean Squared Error) ---
        plot_noise_comparison(
            results_data, visualize_dir, "rmse", 
            ylabel="RMSE (Lower is better)", 
            title="Q Estimation: RMSE", 
            filename="noise_RMSE.png"
        )

        # --- 3. Cosine Similarity ---
        plot_noise_comparison(
            results_data, visualize_dir, "cos", 
            ylabel="Cosine Similarity (Higher is better)", 
            title="Q Estimation: Cosine Similarity", 
            filename="noise_CosSim.png",
            y_limit=[0, 1.1] # 0から1の範囲を強調
        )

        # --- 4. Regret (Log Scale) ---
        plot_noise_comparison(
            results_data, visualize_dir, "Regret", 
            ylabel="Regret (Log scale)", 
            title="Performance: Regret Comparison", 
            filename="noise_Regret.png",
            use_log=True
        )