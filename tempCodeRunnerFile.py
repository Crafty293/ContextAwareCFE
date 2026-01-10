    print("\n===== Final Results =====")
    for i in range(len(results)):
        N = results[i]["N"]
        print(f"N:{N}-----------------")
        for k, v in results[i].items():
            print(f"{k}: {v:.6f}")