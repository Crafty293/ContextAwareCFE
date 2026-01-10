from config import Config
from data_generation import generate_dataset,generate_dataset_with_fixed_Q
import numpy as np
cfg1 = Config(1)
_, Q_1 = generate_dataset(cfg1, 1, cfg1.seed)

cfg2 = Config(10)
_, Q_10 = generate_dataset(cfg2, 10, cfg2.seed)

print(np.allclose(Q_1, Q_10)) # これが True なら完璧！





def compare_datasets(data1, data2):
    print("\n=============== Data Comparison ===============")
    # すべてのキーが一致しているか確認
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    
    if keys1 != keys2:
        print(f"❌ Keys mismatch! \nOnly in 1: {keys1-keys2} \nOnly in 2: {keys2-keys1}")
    
    all_match = True
    for key in keys1:
        if key in data2:
            val1 = data1[key]
            val2 = data2[key]
            
            # 数値の型（NumPy配列など）であれば、全要素が一致するかチェック
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                # 浮動小数点の誤差を考慮して np.allclose を使用
                if np.allclose(val1, val2, atol=1e-8):
                    print(f"✅ {key:10}: Perfectly matches.")
                else:
                    diff = np.abs(val1 - val2).max()
                    print(f"❌ {key:10}: Does NOT match. (Max Diff: {diff:.2e})")
                    all_match = False
            else:
                # 数値以外（リストやスカラ）の比較
                if val1 == val2:
                    print(f"✅ {key:10}: Matches.")
                else:
                    print(f"❌ {key:10}: Does NOT match.")
                    all_match = False
                    
    if all_match:
        print("===============================================")
        print("RESULT: Both datasets are IDENTICAL.")
    else:
        print("===============================================")
        print("RESULT: There are differences between datasets.")

