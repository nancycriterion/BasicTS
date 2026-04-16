import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18          # 👈 全局字体大小设为 18

def draw_gram_matrix():
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    n = 5  # 5x5矩阵
    
    # 创建网格背景
    for i in range(n):
        for j in range(n):
            # 根据位置设置不同颜色
            if i == j:
                color = '#8ECFC9'  # 对角线：青绿色
            elif i > j:
                color = '#FFBE7A'  # 下三角：橙色
            else:
                color = '#FA7F6F'  # 上三角：红色
            
            rect = Rectangle((j, n-1-i), 1, 1, facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            
            # 添加文字标注
            if i == j:
                text = f'$\\cos(2\\phi_{i+1})$'
            else:
                text = f'$\\cos(\\phi_{i+1}+\\phi_{j+1})$'
            
            fontsize = 13 if len(text) > 15 else 13
            ax.text(j+0.5, n-1-i+0.5, text, ha='center', va='center', fontsize=fontsize)
    
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n)+0.5)
    ax.set_yticks(np.arange(n)+0.5)
    ax.set_xticklabels([f'$t_{i+1}$' for i in range(n)], fontsize=18)
    ax.set_yticklabels([f'$t_{i+1}$' for i in range(n)], fontsize=18)
    ax.set_xlabel('时间步 (Key)', fontsize=18)
    ax.set_ylabel('时间步 (Query)', fontsize=18)
    # ax.set_title('格拉姆矩阵 (Gram Matrix) 结构', fontsize=18, fontweight='bold', pad=15)
    ax.set_aspect('equal')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#8ECFC9', edgecolor='black', label='对角线：$\\cos(2\\phi_i)$ (自相关性)'),
        Patch(facecolor='#FFBE7A', edgecolor='black', label='下三角：$\\cos(\\phi_i+\\phi_j)$ (历史相关性)'),
        Patch(facecolor='#FA7F6F', edgecolor='black', label='上三角：$\\cos(\\phi_i+\\phi_j)$ (对称元素)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=15, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(r'D:\xwechat_files\wxid_q3rxpr4sliny22_33e0\msg\file\2026-04\大论文Latex模板（2026）\hnuthesis-hnuthesis-03157e8\figures\gram_matrix_structure.pdf', dpi=300, bbox_inches='tight')
    plt.show()

draw_gram_matrix()

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False


def draw_gasf_flow():
    fig, ax = plt.subplots(1, 1, figsize=(18, 5))
    ax.axis('off')
    # ax.set_title('GASF变换流程', fontsize=18, fontweight='bold', pad=20)
    
    # 定义各阶段的坐标（水平排列）
    steps = [
        {'x': 0.15, 'y': 0.5, 'name': '原始时间序列', 'formula': '$X = [x_1, x_2, ..., x_T]$'},
        {'x': 0.35, 'y': 0.5, 'name': '归一化', 'formula': '$\\widetilde{x_i}=\\frac{x_i}{\\sigma\\cdot2}$'},
        {'x': 0.54, 'y': 0.5, 'name': '极坐标变换', 'formula': '$\\phi_i = \\arccos(\\tilde{x}_i)$'},
        {'x': 0.73, 'y': 0.5, 'name': '格拉姆矩阵', 'formula': '$G_{ij} = \\cos(\\phi_i + \\phi_j)$'},
        {'x': 0.87, 'y': 0.5, 'name': 'GASF图像', 'formula': '$I \\in \\mathbb{R}^{T \\times T}$'}
    ]
    
    colors_step = ['#E8F4F8', '#FFF4E6', '#E8F4F8', '#FFF4E6', '#F0E6F8']
    
    # 绘制流程框
    for i, step in enumerate(steps):
        # 绘制矩形框
        rect = plt.Rectangle((step['x']-0.08, step['y']-0.12), 0.2, 0.24, 
                              facecolor=colors_step[i], edgecolor='black', linewidth=1.5, 
                              hatch=None, zorder=2)
        ax.add_patch(rect)
        
        # 添加名称（粗体）
        ax.text(step['x'], step['y']+0.06, step['name'], 
                ha='center', va='center', fontsize=20
                , fontweight='bold', zorder=3)
        
        # 添加公式（斜体）
        ax.text(step['x'], step['y']-0.05, step['formula'], 
                ha='center', va='center', fontsize=20, style='italic', zorder=3)
    
    # 绘制箭头
    arrow_props = dict(arrowstyle='->', color='gray', lw=2, shrinkA=0, shrinkB=0)
    
    # 箭头1：原始序列 → 归一化
    ax.annotate('', xy=(0.30, 0.5), xytext=(0.25, 0.5), arrowprops=arrow_props)
    
    # 箭头2：归一化 → 极坐标变换
    ax.annotate('', xy=(0.5, 0.5), xytext=(0.45, 0.5), arrowprops=arrow_props)
    
    # 箭头3：极坐标变换 → 格拉姆矩阵
    ax.annotate('', xy=(0.69, 0.5), xytext=(0.64, 0.5), arrowprops=arrow_props)
    
    # 箭头4：格拉姆矩阵 → GASF图像
    ax.annotate('', xy=(0.83, 0.5), xytext=(0.78, 0.5), arrowprops=arrow_props)
    
    # 添加GASF公式说明
    # ax.text(0.5, 0.15, 'GASF = Gramian Angular Summation Field (格拉姆角和场)', 
    #         fontsize=12, ha='center', style='italic',
    #         bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9, edgecolor='gray'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(r'D:\xwechat_files\wxid_q3rxpr4sliny22_33e0\msg\file\2026-04\大论文Latex模板（2026）\hnuthesis-hnuthesis-03157e8\figures\gasf_flowchart.pdf', dpi=300, bbox_inches='tight')
    plt.show()

draw_gasf_flow()