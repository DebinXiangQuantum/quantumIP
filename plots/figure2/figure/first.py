import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import LogLocator  
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
## bar plot legend
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
width_pt = 240
inches_per_pt = 1 / 72.27
fig_width = width_pt * inches_per_pt

# 计算子图尺寸（16:9比例）
subplot_width = fig_width / 2  # 4列
subplot_height = subplot_width *0.6
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
    "lines.markersize": 2.5,
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
orderindex = {}
latencys = {}
thresholdsvals = {}
baselines = ['DV-binary', 'DV-onehot', 'CVQAOA','HybridCVDV']

# 创建模拟数据 - 根据代码需求，需要包含method, cutoff, qubit_width和qumode_width列
methods = ['DV-binary', 'DV-onehot', 'CVQAOA', 'HybridCVDV']
variables = [4, 5, 6, 7, 8, 9]
cutoffs = np.arange(4,20,2)
# 创建数据列表
data = []
for method in methods:
    for variable in variables:
        # 根据不同方法和variable值设置不同的量子比特宽度
        if method == 'DV-binary':
            binary_encoding = np.mean(np.round(np.log2(cutoffs),0))
            qubit_width = variable * binary_encoding
            qumode_width = 0
        elif method == 'DV-onehot':
            onehot_encoding = np.mean(cutoffs)
            qubit_width = variable * onehot_encoding
            qumode_width = 0
        elif method == 'CVQAOA':
            qubit_width = 0
            qumode_width = variable
        else:  # HybridCVDV
            qumode_width = variable
            qubit_width =  variable
        
        
        data.append({
            'method': method,
            'cutoff': variable,
            'variable': variable,
            'qubit_width': qubit_width,
            'qumode_width': qumode_width
        })

# 创建DataFrame
width_data = pd.DataFrame(data)

# 打印数据前几行以验证
print("width_data数据预览:")
print(width_data.head(20))
print("\n数据形状:", width_data.shape)
print("方法列表:", width_data['method'].unique())
print("变量列表:", width_data['variable'].unique())

# Plot threshold vs decoder setting pair
fig, axs = plt.subplots(1, 1, figsize=(fig_width, fig_height))

## set x and y axis limits
labels = ['DV-binary', 'DV-onehot', 'CVQAOA','HybridCVDV']
colors = {'DV-binary': '#FDE725', 'DV-onehot': '#1F9948', 'CVQAOA': '#440154','HybridCVDV': '#1D93D0'}
white_colors = {'DV-binary': '#FDE725', 'DV-onehot': '#1F9948', 'CVQAOA': '#440154','HybridCVDV': '#1D93D0'}
linestyle = {'DV-binary': '-', 'DV-onehot': '--', 'CVQAOA': '-','HybridCVDV': '--'}
markers = {'DV-binary': 's', 'DV-onehot': 's', 'CVQAOA': 'd','HybridCVDV': 'd'}

ax = axs
ax.grid(True,axis='y', linestyle='--')
idxgroup = 0
group_ticks = []
variables = [4,5,6,7,8,9]
for variable in variables:
    idxoff = 0
    for method in width_data['method'].unique():
        data = width_data[(width_data['variable'] == variable) & (width_data['method'] == method)]
        ax.bar(idxgroup + idxoff*2, data['qubit_width'], width=2, color=colors[method], edgecolor='black',  label=method)
        ax.bar(idxgroup + idxoff*2, data['qumode_width'], bottom=data['qubit_width'], width=2, color=colors[method], alpha=0.5, edgecolor='black', label=method)
        idxoff += 1
    group_ticks.append(idxgroup+4)
    idxgroup += 11

ax.set_xticks(np.array(group_ticks)-1)
ax.set_xticklabels(variables)
# ax.set_xticks([5,10,15,20,25],minor=True)
# ax.set_xticks(np.arange(5,100,20),minor=False)
# ax.set_xticklabels(distances)
## draw horizontal line at y=100
ax.set_xlabel(r'Variable', labelpad=0)
ax.set_ylabel('Qubit',labelpad=1)

ax.grid(True,axis='y')
ax.set_ylim(0,90)
ax.set_yticks(np.arange(0,91,30))
ax.set_yticks(np.arange(0,90,10),minor=True)
# ax.set_yscale('log')

legend_handles = [
    Patch(
        facecolor=colors[label],
        edgecolor='black',
        # hatch='//',
        label=label
    )
    for label in labels
]


# 添加全局图例
legend = fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.18), # 调整 y 位置
    framealpha=1,
    ncol=2,                      # 横向排列
    frameon=True,               # 显示边框
    edgecolor='black',        # 边框颜色
    fancybox=False,             # 去掉圆角
    shadow=False,               # 去掉阴影
)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
frame.set_linewidth(0.5)
# # set facecolor to gray
# fig.patch.set_facecolor('#F2F2F2')

plt.savefig('figs/DV-variable.svg', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()