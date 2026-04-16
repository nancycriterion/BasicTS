import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

def plot_causal_mask(seq_len=6, save_path=r'D:\xwechat_files\wxid_q3rxpr4sliny22_33e0\msg\file\2026-04\大论文Latex模板（2026）\hnuthesis-hnuthesis-03157e8\figures\causal_mask.pdf'):
    """
    绘制因果掩码热力图
    行(Query): 当前查询位置
    列(Key): 被关注的位置
    矩阵值: 1表示允许关注，0表示禁止关注
    """
    # 创建因果掩码：下三角(包括对角线)为1，上三角为0
    mask = np.tril(np.ones((seq_len, seq_len)), k=0)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # 热力图：白色=1（允许关注），黑色=0（禁止关注）
    im = ax.imshow(mask, cmap='gray_r', vmin=0, vmax=1, origin='upper')
    
    # 设置坐标轴
    time_labels = [f'第{i+1}步' for i in range(seq_len)]
    ax.set_xticks(np.arange(seq_len))
    ax.set_yticks(np.arange(seq_len))
    ax.set_xticklabels(time_labels, fontsize=18)
    ax.set_yticklabels(time_labels, fontsize=18)
    
    # 添加网格线
    ax.set_xticks(np.arange(seq_len) - 0.5, minor=True)
    ax.set_yticks(np.arange(seq_len) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # 在单元格中显示数值
    for i in range(seq_len):
        for j in range(seq_len):
            ax.text(j, i, f'{int(mask[i, j])}',
                    ha="center", va="center", 
                    color="red" if mask[i, j] == 0 else "w",
                    fontsize=24, fontweight='bold')
    
    ax.set_xlabel('被关注的位置 (Key)', fontsize=20)
    ax.set_ylabel('当前查询位置 (Query)', fontsize=20)
    ax.set_title('下三角(含对角线)=允许(1)，上三角=禁止(0)', fontsize=20)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['禁止 (0)', '允许 (1)'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印矩阵验证
    print("因果掩码矩阵 (行=Query, 列=Key):")
    print(mask)
    return mask

mask = plot_causal_mask(seq_len=6)