import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import font_manager
import os

# 更全面的中文字体设置
def set_chinese_font():
    # 常见的中文字体列表，按优先级排序
    chinese_fonts = [
        'SimHei',           # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'SimSun',           # 宋体
        'KaiTi',            # 楷体
        'FangSong',         # 仿宋
        'STSong',           # 华文宋体
        'STKaiti',          # 华文楷体
        'DejaVu Sans',      # 备用字体
        'Arial Unicode MS'  # 备用字体
    ]

    # 查找系统中可用的中文字体
    available_fonts = []
    for font in chinese_fonts:
        try:
            font_path = font_manager.findfont(font_manager.FontProperties(family=font))
            if font_path and os.path.exists(font_path):
                available_fonts.append(font)
                print(f"找到字体: {font} -> {font_path}")
        except:
            continue

    if available_fonts:
        plt.rcParams['font.sans-serif'] = available_fonts + ['DejaVu Sans', 'Arial Unicode MS']
        print(f"使用字体: {available_fonts[0]}")
    else:
        # 如果找不到中文字体，尝试使用系统字体文件
        try:
            # 常见的中文字体文件路径
            font_paths = [
                '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
                '/usr/share/fonts/truetype/arphic/ukai.ttc',
                '/usr/share/fonts/truetype/arphic/uming.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
            ]
            for path in font_paths:
                if os.path.exists(path):
                    font_prop = font_manager.FontProperties(fname=path)
                    plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'DejaVu Sans']
                    print(f"使用字体文件: {path}")
                    break
        except Exception as e:
            print(f"字体设置失败: {e}")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']

    plt.rcParams['axes.unicode_minus'] = False

# 设置中文字体
set_chinese_font()

# 数据
categories = ['all_correct', 'all_wrong', 'valid']
category_labels = ['all_correct', 'all_wrong', 'valid']  # 更友好的中文标签
counts = [9192, 21572, 63093]

# 创建现代感的配色方案
colors = ['#2E8B57', '#DC143C', '#4169E1']  # 海绿色，深红色，皇家蓝
gradient_colors = ['#4CAF50', '#F44336', '#2196F3']  # 渐变色方案

# 设置样式
sns.set_style("whitegrid")
plt.figure(figsize=(14, 10))

# 创建更美观的条形图
bars = plt.bar(category_labels, counts,
               color=gradient_colors,
               edgecolor='white',
               linewidth=2.5,
               alpha=0.85,
               width=0.7)

# 添加数值标签 - 更精美的样式
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    # 在条形内部显示数值
    plt.text(bar.get_x() + bar.get_width()/2., height - height*0.1,
             f'{count:,}',
             ha='center', va='top',
             fontsize=16,
             fontweight='bold',
             color='white',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.8))

# 设置标题和标签 - 使用字体属性
title_font = {'fontsize': 22, 'fontweight': 'bold', 'color': '#2C3E50'}
label_font = {'fontsize': 16, 'fontweight': 'bold', 'color': '#34495E'}

plt.title('OpenR1-Math-220k\n',
          **title_font, pad=30)

# plt.xlabel('评估类别', **label_font, labelpad=15)
plt.ylabel('number', **label_font, labelpad=15)

# 美化坐标轴
plt.xticks(fontsize=14, fontweight='medium', color='#2C3E50')
plt.yticks(fontsize=12, color='#7F8C8D')

# 设置y轴格式
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# 调整y轴范围，让图形更协调
plt.ylim(0, max(counts) * 1.15)

# 美化网格线
plt.grid(axis='y', alpha=0.4, linestyle='-', color='#BDC3C7')
plt.grid(axis='x', alpha=0)

# 设置背景色
plt.gca().set_facecolor('#F8F9F9')

# 美化边框
for spine in plt.gca().spines.values():
    spine.set_color('#D5DBDB')
    spine.set_linewidth(2)

# 添加百分比标签
total = sum(counts)
percentages = [f'({count/total*100:.1f}%)' for count in counts]

for i, (bar, percentage) in enumerate(zip(bars, percentages)):
    plt.text(bar.get_x() + bar.get_width()/2.,
             bar.get_height() + max(counts)*0.02,
             percentage,
             ha='center',
             va='bottom',
             fontsize=13,
             fontweight='medium',
             color=colors[i],
             style='italic')

# 添加总数据量标注
plt.figtext(0.5, 0.01,
            f'total: {total:,}',
            ha='center',
            fontsize=14,
            fontweight='bold',
            color='#2C3E50',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#ECF0F1', alpha=0.8))

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)

# 保存高质量图片
plt.savefig('/data/home/scyb494/verl/eval/plots/histogram_beautified.png',
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            transparent=False)

plt.show()

print("美化后的直方图已保存为 histogram_beautified.png")

# 打印统计信息
print(f"\n统计信息:")
print(f"总数据量: {total:,}")
for i, (label, count, percentage) in enumerate(zip(category_labels, counts, percentages)):
    print(f"{label}: {count:,} {percentage}")

# 打印当前使用的字体信息
print(f"\n当前使用的字体配置:")
print(f"字体族: {plt.rcParams['font.sans-serif']}")