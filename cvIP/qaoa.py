import numpy as np
import qutip as qt
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize

# --------------------------
# 1. 基础配置：中文显示 + 参数定义
# --------------------------
# 解决中文显示问题
matplotlib.rcParams["font.family"] = ["Heiti TC"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 问题核心参数
N = 7  # Fock空间截断维度
num_modes = 3
p = 2  # QAOA层数（传统分层核心参数）
constraint_coeffs = [3, 1, 1]  # 约束：3n1 + n2 + n3 = 6
target_value = 6
target_operator = None  # 目标函数算符（后续初始化）
null_space_basis = [np.array([1, -3, 0]), np.array([0, 1, -1])]  # 硬约束零空间向量

# 重点跟踪的核心基态（覆盖高/中/低目标值，共6个）
tracked_states = [
    (0, 6, 0),  # 目标值12（理论最优）
    (0, 5, 1),  # 目标值11（次优）
    (0, 4, 2),  # 目标值10（中优）
    (0, 3, 3),  # 目标值9（中优）
    (0, 2, 4),  # 目标值8（中低优）
    (0, 0, 6)   # 目标值6（初始态之一，低优）
]
tracked_labels = [f"|{n1},{n2},{n3}> (t={n1+2*n2+n3})" for n1, n2, n3 in tracked_states]

# --------------------------
# 2. 初始化量子算符（成本H_C + 混合H_M）
# --------------------------
def init_qaoa_hamiltonians():
    global target_operator
    # 基础算符：湮灭/产生/粒子数算符
    a = [qt.tensor([qt.qeye(N) if i != j else qt.destroy(N) for j in range(num_modes)]) 
         for i in range(num_modes)]
    ad = [op.dag() for op in a]
    n = [ad[i] * a[i] for i in range(num_modes)]  # 粒子数算符 n_i = a_i†a_i

    # 成本哈密顿量 H_C（最大化n1+2n2+n3 → 转最小化 -目标函数）
    target_operator = n[0] + 2 * n[1] + n[2]
    H_C = -target_operator

    # 混合哈密顿量 H_M（约束子空间内搅拌）
    H_M = 0 * a[0]
    for u in null_space_basis:
        O_u = 1
        for i in range(num_modes):
            if u[i] > 0:
                O_u *= ad[i] ** u[i]
            elif u[i] < 0:
                O_u *= a[i] ** abs(u[i])
        H_M += (O_u + O_u.dag())

    return H_C, H_M, n

H_C, H_M, n = init_qaoa_hamiltonians()

# --------------------------
# 3. 传统QAOA核心：分层量子电路
# --------------------------
def qaoa_circuit(params, initial_state):
    """分层演化：p层H_C→H_M交替"""
    state = initial_state.copy()
    for i in range(p):
        gamma = params[2*i]    # 第i层H_C参数
        beta = params[2*i + 1] # 第i层H_M参数
        state = qt.propagator(H_C, gamma) * state  # H_C演化
        state = qt.propagator(H_M, beta) * state   # H_M演化
    return state

# --------------------------
# 4. 经典优化：跟踪迭代过程中基态概率（核心新增）
# --------------------------
# 存储迭代过程数据：迭代次数、成本值、各跟踪基态概率
iter_history = {
    "iter": [],          # 迭代序号
    "cost": [],          # 每次迭代的成本值
    "prob": np.zeros((0, len(tracked_states)))  # 每次迭代的基态概率（行：迭代，列：基态）
}

def qaoa_cost_with_history(params, initial_state):
    """带历史跟踪的成本函数：计算成本时，同步记录基态概率"""
    # 记录当前迭代次数（iter_history["iter"]的长度即当前迭代数）
    current_iter = len(iter_history["iter"]) + 1
    iter_history["iter"].append(current_iter)

    # 1. 计算当前参数的成本值
    final_state = qaoa_circuit(params, initial_state)
    cost_val = qt.expect(H_C, final_state)
    iter_history["cost"].append(cost_val)

    # 2. 计算并记录当前迭代中，各跟踪基态的概率
    current_probs = []
    for (n1, n2, n3) in tracked_states:
        basis = qt.tensor(qt.fock(N, n1), qt.fock(N, n2), qt.fock(N, n3))
        prob = abs(basis.overlap(final_state)) ** 2
        current_probs.append(prob)
    iter_history["prob"] = np.vstack([iter_history["prob"], current_probs])

    return cost_val

# 初始化满足约束的初始态
def init_valid_initial_state():
    state1 = qt.tensor(qt.fock(N, 0), qt.fock(N, 6), qt.fock(N, 0))  # |0,6,0>
    state2 = qt.tensor(qt.fock(N, 0), qt.fock(N, 0), qt.fock(N, 6))  # |0,0,6>
    return (state1 + state2).unit()

initial_state = init_valid_initial_state()

# 初始化经典优化参数（2p个随机参数，范围[0, π]）
np.random.seed(42)
initial_params = np.random.uniform(0, np.pi, 2*p)

# 经典优化：带历史跟踪的最小化
optim_result = minimize(
    fun=qaoa_cost_with_history,
    x0=initial_params,
    args=(initial_state,),
    method="COBYLA",
    options={"maxiter": 150, "disp": True}  # 迭代150次：足够观察概率变化趋势
)

optimal_params = optim_result.x
print(f"\n=== 传统QAOA经典优化完成 ===")
print(f"最优分层参数（γ1, β1, γ2, β2）：{optimal_params.round(4)}")
print(f"最小成本值：{optim_result.fun:.6f}")
print(f"总迭代次数：{len(iter_history['iter'])}")

# --------------------------
# 5. 结果分析工具函数
# --------------------------
def get_fock_probs(state):
    probs = np.zeros((N, N, N))
    for n1, n2, n3 in product(range(N), repeat=3):
        basis = qt.tensor(qt.fock(N, n1), qt.fock(N, n2), qt.fock(N, n3))
        probs[n1, n2, n3] = abs(basis.overlap(state)) ** 2
    return probs

def calculate_target_expect(state):
    return qt.expect(target_operator, state)

# --------------------------
# 6. 可视化：重点新增“迭代过程基态概率折线图”
# --------------------------
final_state = qaoa_circuit(optimal_params, initial_state)
final_probs = get_fock_probs(final_state)
final_target_expect = calculate_target_expect(final_state)

# 创建4个子图：迭代概率、成本变化、最终概率、目标值分布
plt.figure(figsize=(16, 12))

# 子图1：核心！每次迭代各基态的概率折线图
plt.subplot(2, 2, 1)
colors = plt.cm.Set3(np.linspace(0, 1, len(tracked_states)))  # 区分度高的颜色
for i in range(len(tracked_states)):
    plt.plot(iter_history["iter"], iter_history["prob"][:, i], 
             linewidth=2.5, label=tracked_labels[i], color=colors[i], marker="o", markersize=3)
plt.xlabel("经典优化迭代次数")
plt.ylabel("基态概率")
plt.title(f"QAOA迭代过程：核心基态概率变化（p={p}层）")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # 图例放右侧，避免遮挡
plt.grid(alpha=0.3)

# 子图2：迭代过程中成本值变化（验证优化收敛性）
plt.subplot(2, 2, 2)
plt.plot(iter_history["iter"], iter_history["cost"], 
         linewidth=3, color="#d62728", marker="s", markersize=4)
plt.xlabel("经典优化迭代次数")
plt.ylabel("成本值（<H_C>）")
plt.title("QAOA迭代过程：成本值收敛趋势")
plt.grid(alpha=0.3)

# 子图3：最终迭代的核心基态概率分布（柱状图）
plt.subplot(2, 2, 3)
final_tracked_probs = [final_probs[n1, n2, n3] for (n1, n2, n3) in tracked_states]
bars = plt.bar(tracked_labels, final_tracked_probs, color=colors)
# 标注最终概率值
for bar, prob in zip(bars, final_tracked_probs):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f"{prob:.3f}", ha="center", va="bottom", fontsize=10)
plt.xlabel("核心基态（带目标值）")
plt.ylabel("最终概率")
plt.title("最终迭代：核心基态概率分布")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)

# 子图4：目标函数期望值与迭代次数关系
plt.subplot(2, 2, 4)
# 计算每次迭代的目标函数期望值（= -成本值）
target_history = [-cost for cost in iter_history["cost"]]
plt.plot(iter_history["iter"], target_history, 
         linewidth=3, color="#2ca02c", marker="^", markersize=4)
plt.axhline(y=12, color="#ff7f0e", linestyle="--", linewidth=2, label="理论最大目标值（12）")
plt.xlabel("经典优化迭代次数")
plt.ylabel("目标函数期望值（n1+2n2+n3）")
plt.title("QAOA迭代过程：目标函数值提升趋势")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figs/qaoa_iteration_prob_history.svg")
plt.show()

# --------------------------
# 7. 最终结果打印
# --------------------------
print(f"\n=== 最终结果详情 ===")
print(f"1. 目标函数期望值：{final_target_expect:.4f}（理论最大值：12）")
print(f"\n2. 迭代过程关键指标变化：")
print(f"   - 初始目标值：{target_history[0]:.4f}")
print(f"   - 最终目标值：{target_history[-1]:.4f}")
print(f"   - 目标值提升幅度：{target_history[-1]-target_history[0]:.4f}")
print(f"\n3. 最终概率最高的3个基态：")
valid_states = [
    ((n1, n2, n3), final_probs[n1, n2, n3], n1+2*n2+n3)
    for n1, n2, n3 in product(range(N), repeat=3)
    if sum(constraint_coeffs[j] * [n1, n2, n3][j] for j in range(num_modes)) == target_value
    and final_probs[n1, n2, n3] > 1e-3
]
valid_states.sort(key=lambda x: x[1], reverse=True)
for i, ((n1, n2, n3), prob, target) in enumerate(valid_states[:3]):
    print(f"   第{i+1}名：|{n1},{n2},{n3}> → 概率={prob:.4f}，目标值={target}")