import pickle
from pathlib import Path
import matplotlib.pyplot as plt

def load_results(pkl_path):
    with open(pkl_path, "rb") as f:
        results_plot = pickle.load(f)
    return results_plot


def plot_Q_error(results_plot, save_dir):
    N_list = [r["N"] for r in results_plot]
    Q_proposed = [r["Q_error"]["proposed"] for r in results_plot]
    Q_cheby = [r["Q_error"]["cheby"] for r in results_plot]

    plt.figure()
    plt.plot(N_list, Q_proposed, marker="o", label="Ref")
    plt.plot(N_list, Q_cheby, marker="s", label="Chebyshev")
    plt.xlabel("N")
    plt.ylabel("Q estimation error")
    plt.title("Q Error(N=1-30)")
    plt.legend()
    plt.grid(True)

    save_path = save_dir / "Q_error.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Q_error plot to {save_path}")


def plot_regret(results_plot, save_dir):
    N_list = [r["N"] for r in results_plot]

    regret_proposed = [r["regret"]["proposed"] for r in results_plot]
    regret_ref = [r["regret"]["ref"] for r in results_plot]
    regret_cheby = [r["regret"]["cheby"] for r in results_plot]
    regret_vi = [r["regret"]["vi"] for r in results_plot]
    regret_euclid = [r["regret"]["euclid"] for r in results_plot]
    regret_maha = [r["regret"]["maha"] for r in results_plot]

    plt.figure()
    plt.plot(N_list, regret_proposed, marker="o", label="Proposed")
    plt.plot(N_list, regret_cheby, marker="s", label="Proposed + Chebyshev")
    plt.plot(N_list, regret_ref, marker="^", label="Reference Q")
    plt.plot(N_list,regret_vi,linestyle="--",label="Proposed vi")
    plt.plot(N_list, regret_euclid, linestyle="--", label="Euclidean baseline")
    plt.plot(N_list, regret_maha, linestyle="--", label="Mahalanobis baseline")

    plt.xlabel("N")
    plt.ylabel("Regret (log scale)")
    plt.yscale("log")
    
    plt.title("Regret N(1-30) (log scale)")
    plt.legend()
    plt.grid(True)

    save_path = save_dir / "regret.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved regret plot to {save_path}")


if __name__ == "__main__":
    # ===== 設定 =====
    result_dir = Path("results")
    visualize_dir = Path("visualize")
    #ディレクトリなかったら作る。
    visualize_dir.mkdir(exist_ok=True)

    # 最新の pkl を自動取得
    pkl_files = sorted(result_dir.glob("exp_*.pkl"))
    if len(pkl_files) == 0:
        raise FileNotFoundError("No experiment result pkl found in results/")

    pkl_path = pkl_files[-1]
    print(f"Loading result file: {pkl_path}")

    # ===== 実行 =====
    results_plot = load_results(pkl_path)
    plot_Q_error(results_plot, visualize_dir)
    plot_regret(results_plot, visualize_dir)
