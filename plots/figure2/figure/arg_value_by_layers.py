import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


width_pt = 240
inches_per_pt = 1 / 72.27
fig_width = width_pt * inches_per_pt

# 计算子图尺寸（16:9比例）
subplot_width = fig_width  # 4列
subplot_height = subplot_width *0.5
fig_height = subplot_height  # 3行
fontsize = 7
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": fontsize,
    "figure.figsize": (fig_width, fig_height),
    "axes.labelsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    ## tick label padding
    "xtick.major.pad": 0.3,
    "ytick.major.pad": 0.3,
    "legend.fontsize": fontsize,
    "axes.titlesize": fontsize,
    "lines.markersize": 4,
    "lines.linewidth": 0.7,
    "lines.markeredgewidth": 0,
    "grid.linewidth": 0.2,
    "grid.alpha": 0.5,
    "grid.color": "gray",
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "hatch.color": "black",
    "hatch.linewidth": 0.5,
})
# 定义文件路径
csv_dir = Path("csv")

# 定义chocoq和qaoa数据文件
chocoq_files = [
    "circuit_analysis_4vars.csv",
    "circuit_analysis_5vars.csv",
    "circuit_analysis_6vars.csv",
    "circuit_analysis_7vars.csv",
    "circuit_analysis_8vars.csv",
    "circuit_analysis_9vars.csv"
]

qaoa_files = [
    "qaoa_results_summary_variables_5.csv",
    "qaoa_results_summary_variables_6.csv",
    "qaoa_results_summary_variables_7.csv",
    "qaoa_results_summary_variables_8.csv",
    "qaoa_results_summary_variables_9.csv",
    "qresults_4_1.csv",
    "qresults_4_2.csv"
]

# 几何平均函数
def geometric_mean(values):
    """计算几何平均值"""
    # 过滤掉非正值，因为几何平均需要正值
    positive_values = values[values > 0]
    if len(positive_values) == 0:
        return np.nan
    return np.exp(np.mean(np.log(positive_values)))

# 处理chocoq数据
chocoq_data = []
for file in chocoq_files:
    file_path = csv_dir / file
    try:
        df = pd.read_csv(file_path)
        # 提取层数和ARG值
        if '层数' in df.columns and 'ARG' in df.columns:
            # 按层数分组并计算几何平均
            layer_groups = df.groupby('层数')['ARG'].apply(geometric_mean).reset_index()
            layer_groups['source'] = file
            layer_groups['type'] = 'chocoq'
            chocoq_data.append(layer_groups)
            print(f"处理chocoq文件 {file}: 找到层数范围 {df['层数'].min()}-{df['层数'].max()}")
        else:
            print(f"警告: 文件 {file} 缺少必要的列")
    except Exception as e:
        print(f"处理文件 {file} 时出错: {e}")

# 处理qaoa数据
qaoa_data = []
for file in qaoa_files:
    file_path = csv_dir / file
    try:
        df = pd.read_csv(file_path)
        # 提取层数和ARG值
        if '层数' in df.columns and 'ARG值' in df.columns:
            # 按层数分组并计算几何平均
            layer_groups = df.groupby('层数')['ARG值'].apply(geometric_mean).reset_index()
            layer_groups['source'] = file
            layer_groups['type'] = 'qaoa'
            qaoa_data.append(layer_groups)
            print(f"处理qaoa文件 {file}: 找到层数范围 {df['层数'].min()}-{df['层数'].max()}")
        else:
            print(f"警告: 文件 {file} 缺少必要的列")
    except Exception as e:
        print(f"处理文件 {file} 时出错: {e}")

bosonic_df = pd.read_csv("csv/bosonicqaoa.csv")
bosonic_by_layer = bosonic_df[bosonic_df['problemid']==1].groupby('p')['ARG'].apply(geometric_mean).reset_index()
bosonic_by_layer['type'] = 'cvqaoa'
print(bosonic_by_layer)
# 合并所有数据
chocoq_df = pd.concat(chocoq_data, ignore_index=True)
qaoa_df = pd.concat(qaoa_data, ignore_index=True)

# 按层数和类型分组，计算所有文件的几何平均
chocoq_by_layer = chocoq_df.groupby('层数')['ARG'].apply(geometric_mean).reset_index()
chocoq_by_layer['type'] = 'chocoq'

qaoa_by_layer = qaoa_df.groupby('层数')['ARG值'].apply(geometric_mean).reset_index()
qaoa_by_layer['type'] = 'qaoa'

# 合并为最终比较数据
comparison_df = pd.concat([
    chocoq_by_layer.rename(columns={'ARG': 'ARG Value'}),
    bosonic_by_layer.rename(columns={'ARG': 'ARG Value','p':'层数'}),
    qaoa_by_layer.rename(columns={'ARG值': 'ARG Value'})
], ignore_index=True)

# 创建可视化图表 - 折线图
plt.figure()

# 绘制折线图
ax = sns.lineplot(
    data=comparison_df, 
    x='层数', 
    y='ARG Value', 
    hue='type',
    style='type',
    markers=True,
    dashes=False,
    palette={'chocoq': '#1F9948',  'cvqaoa':'#440154','qaoa': '#1D93D0',},
)

# 添加数据标签
for line in range(len(comparison_df)):
    x = comparison_df['层数'][line]
    y = comparison_df['ARG Value'][line]
    if y >1:
        ax.text(x, y, f'{y:.0f}', ha='center', va='bottom')
    else:
        ax.text(x, y, f'{y:.2f}', ha='center', va='bottom')

# 设置图表标题和标签
# plt.title('Comparison of Geometric Mean ARG Values by Layers (chocoq vs qaoa)')
plt.xlabel('Number of Layers')
plt.ylabel('Geometric Mean ARG Value (log scale)')
plt.legend(title='Algorithm Type')
plt.tick_params(axis='y', labelcolor='black')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
# 设置Y轴为对数刻度
plt.yscale('log')
plt.ylim(0.01,1e5)

plt.tight_layout()

# 保存图表
output_dir = Path("figs")
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / 'arg_value_by_layers_geometric_mean.svg', format='svg')
plt.close()

# 保存数据到Excel
output_excel = Path("arg_value_by_layers_geometric_mean.xlsx")
with pd.ExcelWriter(output_excel) as writer:
    comparison_df.to_excel(writer, sheet_name='Geometric Mean ARG Comparison', index=False)
    chocoq_df.to_excel(writer, sheet_name='chocoq detailed data', index=False)
    qaoa_df.to_excel(writer, sheet_name='qaoa detailed data', index=False)

print(f"Chart saved to {output_dir / 'arg_value_by_layers_geometric_mean.svg'}")
print(f"Data saved to {output_excel}")

# 打印结果摘要
print("\nGeometric Mean ARG Values Comparison:")
for _, row in comparison_df.iterrows():
    print(f"{row['type']} layer {row['层数']}: {row['ARG Value']:.6f}")