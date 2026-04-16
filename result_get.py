import os
import json
import pandas as pd

root_dir = r"D:\code\big\basicts\BasicTS\checkpoints"

results = []

for model_name in os.listdir(root_dir):
    model_path = os.path.join(root_dir, model_name)

    if not os.path.isdir(model_path):
        continue

    # 第二层：数据集_5_input_output
    for dataset_folder in os.listdir(model_path):
        dataset_path = os.path.join(model_path, dataset_folder)

        if not os.path.isdir(dataset_path):
            continue

        parts = dataset_folder.split("_")

        if len(parts) < 4:
            continue

        # 倒序解析
        output_len = parts[-1]
        input_len = parts[-2]
        dataset_name = "_".join(parts[:-3])

        # 第三层：hash 文件夹
        for hash_folder in os.listdir(dataset_path):
            hash_path = os.path.join(dataset_path, hash_folder)

            if not os.path.isdir(hash_path):
                continue

            json_path = os.path.join(hash_path, "test_metrics.json")

            if not os.path.exists(json_path):
                continue

            with open(json_path, "r") as f:
                data = json.load(f)

            metrics = data.get("overall", {})

            row = {
                "Model": model_name,
                "Dataset": dataset_name,
                "Input Length": int(input_len),
                "Output Length": int(output_len),
                "MAE": round(metrics.get("MAE", 0), 4),
                "MSE": round(metrics.get("MSE", 0), 4),
                "RMSE": round(metrics.get("RMSE", 0), 4),
                "MAPE": round(metrics.get("MAPE", 0), 4),
                "WAPE": round(metrics.get("WAPE", 0), 4),
            }

            results.append(row)

if len(results) == 0:
    print("⚠ 没有读取到任何结果")
else:
    df = pd.DataFrame(results)
    df = df.sort_values(by=["Model", "Dataset", "Input Length", "Output Length"])
    df.to_excel("model_results_summary.xlsx", index=False)
    print("✅ 成功生成 model_results_summary.xlsx")