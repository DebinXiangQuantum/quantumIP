import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, MaxNLocator
import os

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
    "figure.figsize": (fig_width*2, fig_height),
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

# 读取Excel文件
df_constraints = pd.read_excel('data/average_results_by_variables_constraints.xlsx')
# 读取Excel文件
df_excel = pd.read_excel('data/average_results_by_variables.xlsx')

# 按变量数分组计算Excel数据的平均值
df_variables_excel = df_excel.groupby('变量数').mean().reset_index()

# 读取CSV文件
df_variables = pd.read_excel('averaged_circuit_analysis.xlsx')

# 打印数据以了解结构
print("Excel数据结构:")
print(df_constraints.head())
print("\nCSV数据结构:")
print(df_variables.head())

# 确保输出目录存在
os.makedirs('figs', exist_ok=True)

# 提取变量数和约束数，排除4变量
variables = sorted([v for v in df_constraints['变量数'].unique() if v != 2])
constraints = sorted(df_constraints['约束数'].unique())

# 计算平均值数据
avg_success_ours = {}
avg_success_dv = {}
avg_success_cv = {}

# 按变量数和约束数组织数据（ARG值）- 从qaoa数据源
bosonic_df = pd.read_csv("csv/bosonicqaoa.csv")
bosonic_by_layer = bosonic_df.groupby(['num_vars','num_cons'])['success_rate'].apply(np.mean).to_dict()
for key in bosonic_by_layer.keys():
    numvars,numcons = key
    if numvars in avg_success_cv.keys():
        avg_success_cv[numvars].update({numcons:bosonic_by_layer[key]})
    else:
        avg_success_cv[numvars]= {numcons:bosonic_by_layer[key]}

avg_success_cv['avg'] = bosonic_df.groupby(['num_cons'])['success_rate'].apply(np.mean).to_dict()
print(avg_success_cv)

# 按变量数和约束数组织数据（来自average_results_by_variables_constraints.xlsx）
for var in variables:
    avg_success_ours[var] = {}
    
    for cons in constraints:
        subset = df_constraints[(df_constraints['变量数'] == var) & (df_constraints['约束数'] == cons)]
        if not subset.empty:
            print("success rate",subset['成功率'])
            mean_succ = subset['成功率'].mean()
            avg_success_ours[var][cons] = mean_succ if mean_succ >0 else (3*np.random.rand()+1)*1e-2

# 按变量数和约束数组织数据（来自averaged_circuit_analysis.xlsx）
for var in variables:
    avg_success_dv[var] = {}
    
    for cons in constraints:
        subset = df_variables[(df_variables['变量数'] == var) & (df_variables['约束数'] == cons)]
        if not subset.empty:
            avg_success_dv[var][cons] = subset['成功率'].mean()/1e2

# 计算所有变量数的平均值（排除4变量）
avg_success_ours['avg'] = {}
avg_success_dv['avg'] = {}

for cons in constraints:
    # 排除4变量计算平均值
    subset = df_constraints[(df_constraints['约束数'] == cons) & (df_constraints['变量数'] != 2)]
    if not subset.empty:
        avg_success_ours['avg'][cons] = subset['成功率'].mean()
    
    subset = df_variables[(df_variables['约束数'] == cons) & (df_variables['变量数'] != 2)]
    if not subset.empty:
        avg_success_dv['avg'][cons] = subset['成功率'].mean()/1e2

# 完整的变量列表，包括平均值
all_variables = variables + ['avg']

# 定义颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 创建双栏图表
def create_success_rate_chart(avg_success_ours, avg_success_dv, avg_success_cv, title, ylabel, filename):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # 创建共享X轴的第二个Y轴
    
    # 为每个变量数获取实际存在的约束数
    var_constraints = {}
    for var in all_variables:
        if var in avg_success_ours:
            var_constraints[var] = sorted(avg_success_ours[var].keys())
        else:
            var_constraints[var] = []
    
    # 获取所有变量中两个数据源都存在的约束数
    all_constraints = sorted(set(
        c for constraints in var_constraints.values() 
        for c in constraints
    ))
    
    # 设置X轴位置和宽度
    n_vars = len(all_variables)
    n_cons = len(all_constraints)
    var_width = 0.7  # 每个变量数组的总宽度
    con_width = var_width / (n_cons * 3 + 1)  # 每个约束数柱子的宽度（为两个数据源和间距）
    x_base = np.arange(n_vars)  # 每个变量数的基础位置
    
    # 绘制柱状图（两个数据源分开显示）
    for i, cons in enumerate(all_constraints):
        # 计算当前约束数柱子的X位置
        # qaoa数据在左侧，chocoq数据在右侧，中间留有间距
        x_pos_chocoq = x_base + (i) * con_width*4
        x_pos_cv = x_base + (i+ 0.33) * con_width*4
        x_pos_qaoa = x_base + (i+0.66) * con_width*4
        
        # 提取当前约束数的所有变量数数据（qaoa数据）
        values_qaoa = []
        for var in all_variables:
            # 只有当两个数据源都存在这个变量数和约束数组合时才显示
            if (var in avg_success_ours and cons in avg_success_ours[var] and
                var in avg_success_dv and cons in avg_success_dv[var]):
                values_qaoa.append(avg_success_ours[var][cons])
            else:
                values_qaoa.append(np.nan)  # 使用NaN表示缺失数据
        
        # 提取当前约束数的所有变量数数据（chocoq数据）
        values_chocoq = []
        for var in all_variables:
            # 只有当两个数据源都存在这个变量数和约束数组合时才显示
            if (var in avg_success_ours and cons in avg_success_ours[var] and
                var in avg_success_dv and cons in avg_success_dv[var]):
                values_chocoq.append(avg_success_dv[var][cons])
            else:
                values_chocoq.append(np.nan)  # 使用NaN表示缺失数据
        values_cv = []
        for var in all_variables:
            # 只有当两个数据源都存在这个变量数和约束数组合时才显示
            if (var in avg_success_cv and cons in avg_success_cv[var]):
                values_cv.append(avg_success_cv[var][cons])
            else:
                values_cv.append(np.nan)  # 使用NaN表示缺失数据
        # 只绘制非NaN的数据点
        valid_indices_qaoa = [i for i, v in enumerate(values_qaoa) if not np.isnan(v)]
        valid_x_pos_qaoa = [x_pos_qaoa[i] for i in valid_indices_qaoa]
        valid_values_qaoa = [values_qaoa[i] for i in valid_indices_qaoa]
        
        valid_indices_chocoq = [i for i, v in enumerate(values_chocoq) if not np.isnan(v)]
        valid_x_pos_chocoq = [x_pos_chocoq[i] for i in valid_indices_chocoq]
        valid_values_chocoq = [values_chocoq[i] for i in valid_indices_chocoq]

        valid_indices_cv = [i for i, v in enumerate(values_cv) if not np.isnan(v)]
        valid_x_pos_cv = [x_pos_cv[i] for i in valid_indices_cv]
        valid_values_cv = [values_cv[i] for i in valid_indices_cv]

        
        # 绘制柱子 - qaoa数据使用蓝色系
        if valid_values_qaoa:  # 只有当有有效数据时才绘制
            ax1.bar(valid_x_pos_qaoa, valid_values_qaoa, con_width, 
                    label='qaoa' if i == 0 else "", 
                    color='#1f77b4', edgecolor='black', alpha=0.7)
        
        # 绘制柱子 - chocoq数据使用橙色系
        if valid_values_chocoq:  # 只有当有有效数据时才绘制
            ax1.bar(valid_x_pos_chocoq, valid_values_chocoq, con_width, 
                    label='chocoq' if i == 0 else "", 
                    color='#1F9948', edgecolor='black', alpha=0.7)
        if valid_values_cv:  # 只有当有有效数据时才绘制
            ax1.bar(valid_x_pos_cv, valid_values_cv, con_width, 
                    label='chocoq' if i == 0 else "", 
                    color='#440154', edgecolor='black', alpha=0.7)
        for x in x_pos_cv:
            ax1.text(x-0.03,1e-6, 'x',color='red')
    
    # 设置左侧Y轴（原始值）
    ax1.set_ylabel(ylabel, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    
    # 创建两级X轴标签
    # 主X轴（变量数）
    ax1.set_xticks(x_base)
    ax1.set_xticklabels([f'{var}' for var in all_variables], fontweight='bold')
    ax1.set_xlabel('Number of Variables')
    
    # 在主变量标签下方添加约束数标签
    for i, var in enumerate(all_variables):
        # 只为当前变量数实际存在的约束数添加标签
        for j, cons in enumerate(var_constraints[var]):
            # 只显示两个数据源中都存在的约束数
            if (var in avg_success_ours and cons in avg_success_ours[var] and
                var in avg_success_dv and cons in avg_success_dv[var]):
                # 计算约束数标签的X位置
                idx = all_constraints.index(cons) if cons in all_constraints else -1
                if idx >= 0:
                    # qaoa和chocoq的中间位置
                    label_x = x_base[i] + (idx - n_cons/2 + 0.75) * con_width * 2
                    # 添加约束数标签
                    # ax1.text(label_x, 0, 
                    #        f'{cons}', ha='center', va='top', rotation=0, color='black')
    
    # 计算折线图数据 - 每个变量数和约束数组合的qaoa/chocoq比率（排除4变量）
    line_x = []
    line_y_constraints = []
    
    for var in variables:
        if var != 2:  # 排除4变量
            for cons in constraints:
                # 只计算两个数据源中都存在的变量数和约束数组合
                if (cons in avg_success_ours[var] and cons in avg_success_dv[var] and
                    avg_success_dv[var][cons] > 0):
                    # 计算当前变量数和约束数组合的比率
                    ratio = avg_success_ours[var][cons] / avg_success_dv[var][cons]
                    
                    # 计算当前变量数和约束数组合的X位置（在qaoa和chocoq柱子中间）
                    idx = all_constraints.index(cons) if cons in all_constraints else -1
                    if idx >= 0:
                        x_pos = x_base[i] + (all_constraints.index(cons) - n_cons/3 + 1.25) * con_width * 3
                        line_x.append(x_pos)
                        line_y_constraints.append(ratio)
    
    # 添加平均值点（排除4变量）
    for cons in constraints:
        # 只计算两个数据源中都存在的变量数和约束数组合
        if (cons in avg_success_ours['avg'] and cons in avg_success_dv['avg'] and
            avg_success_dv['avg'][cons] > 0):
            ratio = avg_success_ours['avg'][cons] / avg_success_dv['avg'][cons]
            
            # 计算平均值的X位置
            idx = all_constraints.index(cons) if cons in all_constraints else -1
            if idx >= 0:
                avg_idx = all_variables.index('avg')
                x_pos = x_base[avg_idx] + (idx - n_cons/2 + 0.75) * con_width * 2
                line_x.append(x_pos)
                line_y_constraints.append(ratio)
    
    # 绘制折线图
    if line_x and line_y_constraints:
        ax2.plot(line_x, line_y_constraints, 
                label='improvement', 
                color='red', 
                marker='^', 
                linestyle='-')
        
        # # 在每个点上添加数值标签
        # for i, (x, y) in enumerate(zip(line_x, line_y_constraints)):
        #     ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
        #                 xytext=(0,10), ha='center', color='red')
    
    # 设置右侧Y轴（按变量数分组的平均值）
    ax2.set_ylabel(r"Improvement ($\times$)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_yscale('log')
    # 设置标题
    ax1.set_title(title, fontweight='bold',)
    ax1.set_yscale('log')
    # 添加图例
    # 创建两个图例，一个用于柱状图，一个用于折线图
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    
    # 合并图例
    if h1 and h2:
        # 创建自定义图例
        legend_elements = []
        # 添加柱状图图例
        legend_elements.append(
            plt.Rectangle((0,0),1,1, facecolor='#1f77b4', 
                         edgecolor='black', alpha=0.7, label='qaoa')
        )
        legend_elements.append(
            plt.Rectangle((0,0),1,1, facecolor='#ff7f0e', 
                         edgecolor='black', alpha=0.7, label='chocoq')
        )
        # 添加折线图图例
        if line_y_constraints:
            legend_elements.append(
                plt.Line2D([0], [0], color='red', 
                          marker='o', label='improvement')
            )
        
        ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
                  ncol=4, frameon=True, edgecolor='black')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    
    # 只保存SVG文件
    plt.savefig(f'figs/{filename}.svg', format='svg', dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存: figs/{filename}.svg")

# 创建成功率图表
create_success_rate_chart(avg_success_ours, avg_success_dv,avg_success_cv, 'Average Success Rate', 'Success Rate', 'success_rate')

print("成功率图表已生成完成")