import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
import gc
from collections import defaultdict

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20       # 👈 全局字体大小设为 18


base_path = Path(r"D:\code\big\basicts\BasicTS\checkpoints")
output_path = Path(r"D:\code\big\basicts\BasicTS\comparison_plots0416")
output_path.mkdir(parents=True, exist_ok=True)

models = [
    "Crossformer", "FreTS", "iTransformerForForecasting",
    "myModel", #"myModel1", "myModel2", "myModel3",
    "NLinear", "SOFTS", "SparseTSF", "TimeMixerForForecasting", "TimeXer"
]

def parse_dataset_info(folder_name):
    """解析文件夹名称，格式如: BeijingAirQuality_5_96_96 或 illness_5_36_60"""
    parts = folder_name.split('_')
    
    if len(parts) >= 4:
        try:
            n_features = int(parts[-3])
            input_len = int(parts[-2])
            pred_len = int(parts[-1])
            dataset_name = '_'.join(parts[:-3])
            return dataset_name, n_features, input_len, pred_len
        except:
            pass
    
    return folder_name, None, None, None

def load_binary_data(file_path, max_points=200):
    """直接读取二进制文件为float32"""
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        data = np.frombuffer(raw_data, dtype=np.float32)
        
        if len(data) == 0:
            return None
        
        if len(data) > max_points:
            return data[:max_points]
        return data
        
    except Exception as e:
        return None

def find_all_test_results():
    """查找所有test_results文件夹"""
    all_folders = []
    
    print("正在扫描文件夹...")
    for model in models:
        model_path = base_path / model
        if not model_path.exists():
            continue
        
        test_dirs = list(model_path.rglob("test_results"))
        for test_dir in test_dirs:
            pred_file = test_dir / "prediction.npy"
            target_file = test_dir / "targets.npy"
            
            if pred_file.exists() and target_file.exists():
                # 获取数据集文件夹名称（test_results的父文件夹）
                dataset_folder = test_dir.parent.name
                
                # 如果父文件夹是hash值，再取上一层
                if len(dataset_folder) == 32 and all(c in '0123456789abcdef' for c in dataset_folder):
                    dataset_folder = test_dir.parent.parent.name
                
                # 解析数据集信息
                dataset_name, n_features, input_len, pred_len = parse_dataset_info(dataset_folder)
                
                all_folders.append({
                    'model': model,
                    'dataset_folder': dataset_folder,
                    'dataset_name': dataset_name,
                    'n_features': n_features,
                    'input_len': input_len,
                    'pred_len': pred_len,
                    'path': test_dir
                })
    
    return all_folders

def plot_all_models_on_one_figure(dataset_name, pred_len, folders, feature_idx=0):
    """将所有模型的预测值和真实值画在一张图上"""
    if not folders:
        return
    
    # 构建文件名
    if pred_len:
        filename = f"{dataset_name}_pred{pred_len}"
    else:
        filename = dataset_name
    
    print(f"  绘制: {filename} - 特征{feature_idx+1} (共{len(folders)}个模型)")
    
    plt.figure(figsize=(14, 7))
    
    # 加载真实值
    targets = None
    if folders:
        target_file = folders[0]['path'] / "targets.npy"
        targets_raw = load_binary_data(target_file, max_points=200)
        if targets_raw is not None:
            targets = targets_raw
    
    # 使用不同颜色绘制各模型的预测值
    colors = plt.cm.tab20(np.linspace(0, 1, len(folders)))
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    valid_models = []
    
    for idx, folder_info in enumerate(folders):
        pred_file = folder_info['path'] / "prediction.npy"
        predictions = load_binary_data(pred_file, max_points=200)
        
        if predictions is not None and len(predictions) > 0:
            valid_models.append(folder_info['model'])
            x = np.arange(len(predictions))
            
            color = colors[idx % len(colors)]
            linestyle = line_styles[idx % len(line_styles)]
            
            plt.plot(x, predictions, 
                    color=color, 
                    linestyle=linestyle,
                    label=folder_info['model'], 
                    linewidth=1.5, 
                    alpha=0.8)
    
    # 绘制真实值
    if targets is not None and len(targets) > 0:
        x = np.arange(len(targets))
        plt.plot(x, targets, 'k-', label='真实值', 
                linewidth=2.5, alpha=0.95, zorder=100)
    
    plt.xlabel('时间步', fontsize=20)
    plt.ylabel('值', fontsize=20)
    # plt.legend(loc='best', fontsize=18, frameon=True, ncol=2)
    
    plt.legend(loc='center left',fontsize=18, frameon=True, bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    # plt.show()
    
    # 保存文件
    save_filename = f"{filename}_feature{feature_idx+1}.pdf"
    save_path = output_path / save_filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    已保存: {save_filename} (模型: {', '.join(valid_models)})")

def main():
    print("="*60)
    print("多维时间序列预测结果对比")
    print("="*60)
    
    all_folders = find_all_test_results()
    
    if not all_folders:
        print("没有找到任何test_results文件夹！")
        return
    
    print(f"\n找到 {len(all_folders)} 个预测结果文件夹")
    
    # 按数据集名称和预测长度分组
    grouped = defaultdict(list)
    for folder in all_folders:
        key = (folder['dataset_name'], folder['pred_len'])
        grouped[key].append(folder)
    
    print(f"\n共 {len(grouped)} 个不同的数据集配置\n")
    
    # 显示分组信息
    print("数据集配置列表:")
    dataset_list = list(grouped.items())
    for i, ((dataset_name, pred_len), folders) in enumerate(dataset_list, 1):
        if pred_len:
            info = f"预测长度={pred_len}"
        else:
            info = "预测长度未知"
        print(f"  {i}. {dataset_name} - {info} ({len(folders)} 个模型)")
    
    # 让用户选择要绘制的特征
    print(f"\n请选择要绘制的特征索引 (0开始，默认0): ", end="")
    feature_choice = input().strip()
    feature_idx = int(feature_choice) if feature_choice else 0
    
    # 确认是否绘制所有特征
    print(f"是否绘制所有特征？(y/n，默认n): ", end="")
    plot_all = input().strip().lower() == 'y'
    
    print("\n开始绘图...")
    print("-"*60)
    
    for idx, ((dataset_name, pred_len), folders) in enumerate(dataset_list, 1):
        # 获取特征数
        n_features = 1
        # print(dataset_name)
        if dataset_name not in ['Weather','ETTm2','ExchangeRate','BeijingAirQuality'] or pred_len not in [96,336]:
            continue
        
        for f in folders:
            if f['n_features']:
                n_features = f['n_features']
                break
        
        if plot_all and n_features > 1:
            print(f"\n[{idx}/{len(grouped)}] {dataset_name} (预测长度={pred_len}) - 共 {n_features} 个特征")
            for f_idx in range(n_features):
                plot_all_models_on_one_figure(dataset_name, pred_len, folders, f_idx)
                gc.collect()
        else:
            print(f"\n[{idx}/{len(grouped)}] 处理: {dataset_name} (预测长度={pred_len})")
            plot_all_models_on_one_figure(dataset_name, pred_len, folders, feature_idx)
            gc.collect()
    
    print("\n" + "="*60)
    print(f"完成！所有图表已保存到: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()