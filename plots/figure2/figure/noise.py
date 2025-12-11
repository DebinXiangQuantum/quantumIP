import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from scipy.interpolate import make_interp_spline

# --- 数据准备 ---
data = """Time ttt,Raw HCH_CHC​,Raw VVV,Joint GSE HCH_CHC​ (K=4),Joint GSE VVV,V-First GSE HCH_CHC​ (K=4),V-First GSE VVV,Dual-GSE Joint HCH_CHC​ (K=1),Dual-GSE Joint VVV
0.00,-2.000,0.000,-2.000,0.000,-2.000,0.000,-2.000,0.000
0.03,-1.900,0.064,-1.921,0.000,-1.921,0.000,-1.900,0.000
0.06,-1.813,0.125,-1.848,0.000,-1.848,0.000,-1.813,0.000
0.10,-1.737,0.182,-2.064,2.218,-1.784,0.000,-1.737,0.182
0.13,-1.675,0.237,-2.067,2.171,-1.729,0.000,-1.675,0.237
0.16,-1.625,0.289,-2.069,2.112,-1.684,0.000,-1.625,0.289
0.19,-1.588,0.339,-2.073,2.054,-1.650,0.000,-1.588,0.339
0.22,-1.562,0.387,-2.077,2.000,-1.626,0.000,-1.562,0.387
0.25,-1.549,0.433,-2.081,1.947,-1.612,0.000,-1.549,0.433
0.28,-1.545,0.478,-2.086,1.896,-1.607,0.000,-1.545,0.478
MSE,0.057,0.057,0.006,1.948,0.019,0.000,0.057,0.057"""

# 读取数据
df = pd.read_csv(io.StringIO(data))

# 清洗数据：删除 'MSE' 行，并确保 Time 列是数值型
df_clean = df[df['Time ttt'] != 'MSE'].copy()
df_clean['Time ttt'] = pd.to_numeric(df_clean['Time ttt'])

# 获取原始的 X 轴数据（时间）
x_original = df_clean['Time ttt'].values

# --- 关键步骤：创建用于平滑的密集 X 轴 ---
# 在原始时间的最小值和最大值之间生成 300 个密集的点
x_smooth = np.linspace(x_original.min(), x_original.max(), 300)

# --- 绘图 ---
plt.figure(figsize=(12, 7))

# 设置颜色循环，确保散点和曲线颜色一致
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# 循环处理每一列数据
for i, column in enumerate(df_clean.columns[1:]):
    y_original = df_clean[column].values
    color = colors[i % len(colors)] # 获取当前颜色

    # 1. 创建样条插值模型 (k=3 表示三次样条，这是最常用的平滑方法)
    spline_model = make_interp_spline(x_original, y_original, k=3)
    
    # 2. 计算平滑后的 Y 值
    y_smooth = spline_model(x_smooth)

    # 3. 绘制平滑曲线
    # alpha=0.8 让线稍微透明一点点
    plt.plot(x_smooth, y_smooth, label=column, color=color, linewidth=2, alpha=0.8)
    
    # 4. (可选) 将原始数据点以散点的形式标出
    # 这样既能看到平滑趋势，又能知道真实数据在哪里
    plt.scatter(x_original, y_original, color=color, s=30, marker='o', alpha=0.6)

# 图表修饰
plt.title('Time Series Data (Smoothed)', fontsize=14)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('Value', fontsize=12)
# 将图例移到图外右侧，防止遮挡
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("noise.svg")
# 显示图表
plt.show()