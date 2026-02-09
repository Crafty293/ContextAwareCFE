import pickle
from pathlib import Path
import matplotlib.pyplot as plt

def load_results(pkl_path):
    with open(pkl_path, "rb") as f:
        results_plot = pickle.load(f)
    return results_plot

def plot_Q_errors(results_plot, save_dir):
    """MAE, RMSE, Cosine Similarityをそれぞれプロット"""
    N_list = [r["N"] for r in results_plot]
    metrics = ["mae", "rmse", "cos"]
    
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        
        # 各手法のデータを抽出
        # 実験コードの構造：res[metric]["proposed"]
        val_prop = [r[metric]["proposed"] for r in results_plot]
        val_cheby = [r[metric]["cheby"] for r in results_plot]
        val_ref = [r[metric]["ref"] for r in results_plot]

        plt.plot(N_list, val_prop, marker="o", label="Proposed-ref")
        plt.plot(N_list, val_cheby, marker="s", label="Proposed-Chebyshev")
        plt.plot(N_list, val_ref, marker="x", linestyle=":", label="Random Q")

        plt.xlabel("Sample Size N")
        plt.ylabel(f"Q estimation error ({metric.upper()})")
        plt.title(f"Q Error Comparison: {metric.upper()} (N=1-30)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        save_path = save_dir / f"Q_error_{metric}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {metric} plot to {save_path}")

def plot_regret(results_plot, save_dir):
    """Regretをプロット（Baselineとの比較）"""
    N_list = [r["N"] for r in results_plot]

    # 実験コードの構造：res["Regret"]["proposed"]
    regret_prop = [r["Regret"]["proposed"] for r in results_plot]
    regret_cheby = [r["Regret"]["cheby"] for r in results_plot]
    regret_ref = [r["Regret"]["ref"] for r in results_plot]
    regret_euclid = [r["Regret"]["euclid"] for r in results_plot]
    regret_maha = [r["Regret"]["maha"] for r in results_plot]

    plt.figure(figsize=(10, 6))
    
    # 提案・改良手法
    plt.plot(N_list, regret_prop, marker="o", label="Proposed-ref", linewidth=2)
    plt.plot(N_list, regret_cheby, marker="s", label="Proposed-Chebyshev", linewidth=2)
    
    # 比較対象
    plt.plot(N_list, regret_ref, marker="x", linestyle=":", label="Random Q")
    plt.plot(N_list, regret_euclid, linestyle="--", label="Euclidean baseline", alpha=0.8)
    plt.plot(N_list, regret_maha, linestyle="--", label="Mahalanobis baseline", alpha=0.8)

    plt.xlabel("Sample Size N")
    plt.ylabel("Regret (log scale)")
    plt.yscale("log") # 実験コードの要望通りログスケール
    
    plt.title("Regret Analysis (N=1-30)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 凡例を外に出して見やすく
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    save_path = save_dir / "regret_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved regret plot to {save_path}")

if __name__ == "__main__":
    # ===== 設定 =====
    result_dir = Path("results")
    visualize_dir = Path("visualize")
    visualize_dir.mkdir(exist_ok=True)

    # 最新の pkl を自動取得
    pkl_files = sorted(result_dir.glob("exp_no_noise_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError("No experiment result pkl found in results/")

    pkl_path = pkl_files[-1]
    print(f"Loading result file: {pkl_path}")

    # ===== 実行 =====
    results_data = load_results(pkl_path)
    
    # 1. Qの推定誤差 (MAE, RMSE, COS) のグラフ作成
    plot_Q_errors(results_data, visualize_dir)
    
    # 2. Regret のグラフ作成
    plot_regret(results_data, visualize_dir)
