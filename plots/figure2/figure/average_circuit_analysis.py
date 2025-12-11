import pandas as pd
import numpy as np
import os

# 定义CSV文件路径
csv_files = [
    'csv/circuit_analysis_4vars.csv',
    'csv/circuit_analysis_5vars.csv',
    'csv/circuit_analysis_6vars.csv',
    'csv/circuit_analysis_7vars.csv',
    'csv/circuit_analysis_8vars.csv',
    'csv/circuit_analysis_9vars.csv'
]

# 创建一个空列表来存储所有数据
all_data = []

# 读取每个CSV文件并提取所需数据
for csv_file in csv_files:
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        # 提取变量数
        var_num = int(csv_file.split('_')[-1].split('.')[0].replace('vars', ''))
        
        # 提取需要的列：成功率、约束满足率、ARG值
        # 根据列名提取数据
        for _, row in df.iterrows():
            # 确保所有需要的列都存在
            if '最佳解概率(%)' in df.columns and '满足约束概率(%)' in df.columns and 'ARG' in df.columns:
                all_data.append({
                    '变量数': var_num,
                    '约束数': row['约束数'],
                    '成功率': row['最佳解概率(%)'] / 100.0,  # 转换为小数
                    '约束满足率': row['满足约束概率(%)'] / 100.0,  # 转换为小数
                    'ARG值': row['ARG']
                })

# 将数据转换为DataFrame
result_df = pd.DataFrame(all_data)

# 按变量数和约束数分组并计算平均值
averaged_data = result_df.groupby(['变量数', '约束数']).agg({
    '成功率': 'mean',
    '约束满足率': 'mean',
    'ARG值': 'mean'
}).reset_index()

# 保存到新的Excel文件
output_file = 'figure/averaged_circuit_analysis.xlsx'
averaged_data.to_excel(output_file, index=False)

print(f"按变量数和约束数分组的平均数据已保存到: {output_file}")
print(averaged_data.head())

# 按变量数分组并计算平均值
averaged_by_vars = result_df.groupby(['变量数']).agg({
    '成功率': 'mean',
    '约束满足率': 'mean',
    'ARG值': 'mean'
}).reset_index()

# 保存到新的Excel文件
output_file_vars = 'figure/averaged_circuit_analysis_by_vars.xlsx'
averaged_by_vars.to_excel(output_file_vars, index=False)

print(f"\n按变量数分组的平均数据已保存到: {output_file_vars}")
print(averaged_by_vars)