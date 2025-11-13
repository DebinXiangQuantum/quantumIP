# --- 11. 主程序 ---
from tree_compiler.compiler import Hamiltonian, find_optimal_path, reconstruct_path
from tree_compiler.operator import a, a_dag
if __name__ == "__main__":
    # 示例：创建一个简单的哈密顿量
    # H = a(1) * a_dag(2)**3 + a_dag(2)**3 * a_dag(1)
    # 这里简化示例
    # 创建一个测试哈密顿量
    # 注意：由于 operator.py 末尾有示例代码会执行，我们需要确保导入正确
    H_expr = a(1) * a_dag(2)**3 + a(2)**3 * a_dag(1)
    root_H = Hamiltonian(expr=H_expr, is_leaf=False)
    
    print(f"开始搜索，根哈密顿量: {root_H}")
    
    # 运行 A* 搜索
    optimal_node = find_optimal_path(root_H)
    
    # 打印结果
    if optimal_node:
        reconstruct_path(optimal_node)
    else:
        print("未找到最优路径。")