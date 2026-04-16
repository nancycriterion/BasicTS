import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

def draw_mlp_vs_kan_compact():
    # 创建画布，整体尺寸不变，但内容更紧凑
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ========== 左图：MLP架构 ==========
    ax1 = axes[0]
    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(0, 3.5)
    ax1.axis('off')
    # ax1.set_title('(a) 多层感知机 (MLP)', fontsize=18, fontweight='bold', pad=15)
    
    # 层配置：输入2，隐藏3，输出1
    layers_mlp = [2, 3, 1]
    colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F']
    layer_x = [0, 1.2, 2.4]  # x坐标紧凑排列
    
    # 存储节点位置
    nodes_mlp = {0: [], 1: [], 2: []}
    
    for l, n_nodes in enumerate(layers_mlp):
        # y轴范围压缩，节点更集中
        if n_nodes == 1:
            y_pos = [1.75]
        elif n_nodes == 2:
            y_pos = [1.2, 2.3]
        else:  # 3个节点
            y_pos = [1.0, 1.9, 2.8]
        
        for i, y in enumerate(y_pos):
            # 绘制节点（稍大一点）
            circle = plt.Circle((layer_x[l], y), 0.15, color=colors[l], ec='black', linewidth=1.5, zorder=2)
            ax1.add_patch(circle)
            nodes_mlp[l].append((layer_x[l], y))
            
            # 添加标签
            if l == 0:
                ax1.text(layer_x[l]-0.2, y, f'$x_{i+1}$', fontsize=17, ha='right', va='center')
            elif l == 2:
                ax1.text(layer_x[l]+0.2, y, f'$\\hat{{y}}$', fontsize=17, ha='left', va='center')
            else:
                ax1.text(layer_x[l], y+0.22, f'$h_{i+1}^{{(1)}}$', fontsize=17, ha='center')
    
    # 绘制连接线
    for l in range(len(layers_mlp)-1):
        for (x1, y1) in nodes_mlp[l]:
            for (x2, y2) in nodes_mlp[l+1]:
                ax1.plot([x1+0.15, x2-0.15], [y1, y2], 'gray', linewidth=1, alpha=0.5, zorder=1)
    
    # 添加标注
    ax1.annotate('MLP固定激活函数', xy=(1.2, 0.5), xytext=(1.2, -0.2),
                 fontsize=20, ha='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax1.text(1.2, 0.9, '$\\sigma(\\cdot)$', fontsize=18, ha='center', color='#D62728')
    
    # ========== 右图：KAN架构 ==========
    ax2 = axes[1]
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(0, 3.5)
    ax2.axis('off')
    # ax2.set_title('(b) Kolmogorov-Arnold网络 (KAN)', fontsize=18, fontweight='bold', pad=10)
    
    layers_kan = [2, 3, 1]
    nodes_kan = {0: [], 1: [], 2: []}
    
    for l, n_nodes in enumerate(layers_kan):
        if n_nodes == 1:
            y_pos = [1.75]
        elif n_nodes == 2:
            y_pos = [1.2, 2.3]
        else:
            y_pos = [1.0, 1.9, 2.8]
        
        for i, y in enumerate(y_pos):
            # KAN节点：白色填充，黑色边框
            circle = plt.Circle((layer_x[l], y), 0.15, color='white', ec='black', linewidth=1.8, zorder=2)
            ax2.add_patch(circle)
            nodes_kan[l].append((layer_x[l], y))
            
            if l == 0:
                ax2.text(layer_x[l]-0.2, y, f'$x_{i+1}$', fontsize=17, ha='right', va='center')
            elif l == 2:
                ax2.text(layer_x[l]+0.2, y, f'$\\hat{{y}}$', fontsize=17, ha='left', va='center')
            else:
                ax2.text(layer_x[l], y+0.22, f'$h_{i+1}^{{(1)}}$', fontsize=17, ha='center')
    
    # 绘制边（带可学习函数标注）
    for l in range(len(layers_kan)-1):
        for idx_i, (x1, y1) in enumerate(nodes_kan[l]):
            for idx_j, (x2, y2) in enumerate(nodes_kan[l+1]):
                # 使用渐变色线条
                ax2.plot([x1+0.15, x2-0.15], [y1, y2], '#FFBE7A', linewidth=1.5, alpha=0.8, zorder=1)
                # 在边中点添加函数标注
                mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
                ax2.text(mid_x, mid_y+0.1, f'$\\phi_{{{idx_j+1},{idx_i+1}}}$', 
                         fontsize=13, ha='center', color='#D62728', fontweight='bold')
    
    # 添加标注
    ax2.annotate('KAN可学习激活函数', xy=(1.2, 0.5), xytext=(1.2, -0.2),
                 fontsize=20, ha='center',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    ax2.text(1.2, 0.9, '$\\phi(\\cdot)$', fontsize=18, ha='center', color='#D62728')
    
    # # 添加整体图注说明
    # fig.text(0.5, 0.02, 'MLP在节点上放置固定激活函数，KAN在边上放置可学习激活函数', 
    #          fontsize=12, ha='center', va='center',
    #          bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9))
    
    # plt.suptitle('MLP与KAN架构对比', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1)
    plt.savefig(r'D:\xwechat_files\wxid_q3rxpr4sliny22_33e0\msg\file\2026-04\大论文Latex模板（2026）\hnuthesis-hnuthesis-03157e8\figures\mlp_vs_kan_compact.pdf', dpi=300, bbox_inches='tight')
    plt.show()

draw_mlp_vs_kan_compact()