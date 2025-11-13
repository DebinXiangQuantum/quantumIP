import heapq
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Callable, FrozenSet, Optional
# 导入本地 operator.py 中的符号算符系统
from .operator import (
    SymbolicOperator, SumOp, ProductOp, Scalar, BosonicOp, PauliOp,GateOp,
    CommutatorOp, AntiCommutatorOp, BOp, X, Y, Z, I, a, a_dag, extract_qumodes
)
import math

def dagger(op: SymbolicOperator) -> SymbolicOperator:
    """Compute the adjoint (dagger) of the operator."""
    if isinstance(op, Scalar):
        return Scalar(op.value.conjugate())
    elif isinstance(op, PauliOp):
        return op  # Paulis are Hermitian
    elif isinstance(op, BosonicOp):
        return BosonicOp(not op.is_creation, op.index)
    elif isinstance(op, SumOp):
        return SumOp([dagger(term) for term in op.terms])
    elif isinstance(op, ProductOp):
        return ProductOp([dagger(f) for f in reversed(op.factors)])
    elif isinstance(op, CommutatorOp):
        return CommutatorOp(dagger(op.B), dagger(op.A)) * Scalar(-1)
    elif isinstance(op, AntiCommutatorOp):
        return AntiCommutatorOp(dagger(op.A), dagger(op.B))
    elif isinstance(op, GateOp):
        return op.dagger()
    elif isinstance(op, BOp):
        return BOp(op.A, op.alpha.conjugate())
    else:
        raise ValueError(f"Cannot compute dagger for {type(op)}")

# --- 1. 硬件和代价定义 ---
T_SINGLE = 20  # ns (单 qumode 门)
T_MULTI = 700  # ns (多 qumode 门，取 400-1000ns 的平均值)

# --- 2. 基于 SymbolicOperator 的 Hamiltonian 包装类 ---
@dataclass(frozen=True)
class Hamiltonian:
    """
    基于 SymbolicOperator 的哈密顿量包装类，用于 A* 搜索。
    包含符号表达式、qumode 集合、执行代价等信息。
    """
    expr: SymbolicOperator  # 符号表达式
    qumodes: FrozenSet[int] = field(init=False)  # 涉及的 Qumode 集合
    cost: int = 0  # 基础门的执行时间 (0 表示非基础门)
    is_leaf: bool = False  # 这是否是一个基础门 (叶节点)
    conditions: Dict[int, int] = field(default_factory=dict)  # qubit_index: initial_state (0 or 1)
    
    def __post_init__(self):
        # 计算 qumodes 集合
        qumodes_set = extract_qumodes(self.expr)
        object.__setattr__(self, 'qumodes', frozenset(qumodes_set))
        
        # 如果是叶节点，计算代价
        if self.is_leaf:
            # 单 qumode 门 vs 多 qumode 门
            if len(self.qumodes) <= 1:
                object.__setattr__(self, 'cost', T_SINGLE)
            else:
                object.__setattr__(self, 'cost', T_MULTI)
    
    def __repr__(self):
        cond_str = f", conditions={self.conditions}" if self.conditions else ""
        return f"H({self.expr}{cond_str})"
    
    def __eq__(self, other):
        if not isinstance(other, Hamiltonian):
            return False
        return (self.expr == other.expr and
                self.qumodes == other.qumodes and
                self.cost == other.cost and
                self.is_leaf == other.is_leaf and
                self.conditions == other.conditions)
    
    def __hash__(self):
        cond_tuple = tuple(sorted(self.conditions.items()))
        return hash((hash(self.expr), hash(self.qumodes), self.cost, self.is_leaf, cond_tuple))

# --- 3. 搜索节点 (Node) 的定义 ---
@dataclass(order=True)
class SearchNode:
    """定义 A* 搜索树中的一个节点。"""
    
    f_cost: float = field(init=False)
    g_cost: float = field(compare=False)
    
    # 状态：一个由"并行块"组成的列表
    blocks: Tuple[FrozenSet[Hamiltonian], ...] = field(compare=False)
    
    # 用于回溯路径
    parent: 'SearchNode' = field(default=None, compare=False)
    rule_applied: str = field(default="Start", compare=False)

    def __post_init__(self):
        # f = g + h
        self.f_cost = self.g_cost + self.calculate_h_cost()

    def calculate_h_cost(self) -> float:
        """启发式函数 (h-cost): 估计剩余代价。"""
        h = 0.0
        for block in self.blocks:
            for H in block:
                if not H.is_leaf:
                    # 基于表达式复杂度估计剩余代价
                    h += estimate_remaining_cost(H)
        return h

    def __hash__(self):
        return hash(self.blocks)

    def __eq__(self, other):
        if not isinstance(other, SearchNode):
            return False
        return self.blocks == other.blocks

# --- 4. 对易性检查 ---

def check_commutativity(H1: Hamiltonian, H2: Hamiltonian) -> bool:
    """
    检查两个哈密顿量是否对易。
    简化实现：如果它们在不相交的 qumode 上操作，则它们对易。
    更复杂的对易性检查需要符号计算，这里使用简化版本。
    """
    return H1.qumodes.isdisjoint(H2.qumodes)

def check_commutativity_for_block(H: Hamiltonian, block: FrozenSet[Hamiltonian]) -> bool:
    """检查一个 H 是否与一个块中的所有其他 H 都对易。"""
    return all(check_commutativity(H, H_other) for H_other in block)

# --- 5. 压缩逻辑 ---

def compact(hamiltonians: List[Hamiltonian]) -> Tuple[FrozenSet[Hamiltonian], ...]:
    """
    核心压缩逻辑：将一个扁平的 H 列表压缩成并行的块。
    使用贪心"向左合并"策略。
    """
    if not hamiltonians:
        return tuple()

    blocks_list: List[Set[Hamiltonian]] = []
    assert isinstance(hamiltonians, list)
    for H in hamiltonians:
        merged = False
        # 尝试合并到最后一个（即邻近的）块中
        if blocks_list and check_commutativity_for_block(H, blocks_list[-1]):
            blocks_list[-1].add(H)
            merged = True
        
        if not merged:
            # 无法合并，创建新块
            blocks_list.append({H})
            
    # 转换为不可变的元组和 frozensets，使其可哈希
    return tuple(frozenset(block) for block in blocks_list)

def calculate_g_cost(blocks: Tuple[FrozenSet[Hamiltonian], ...]) -> float:
    """
    (g-cost) 计算一个节点（块序列）的总执行时间。
    """
    total_cost = 0.0
    for block in blocks:
        # 块的代价 = 块中代价最大的那个门
        if block:
            block_cost = max(H.cost for H in block)
            total_cost += block_cost
    return total_cost

# --- 6. 代价估计函数 ---

def estimate_remaining_cost(H: Hamiltonian) -> float:
    """
    估计分解一个非叶节点哈密顿量所需的最小时间。
    这是一个启发式函数，必须"可接受的"（不能高估）。
    """
    # 基于表达式的复杂度进行估计
    expr = H.expr
    
    # 如果涉及多个 qumode，可能需要多 qumode 门
    num_qumodes = len(H.qumodes)
    
    if num_qumodes == 0:
        return 0.0
    elif num_qumodes == 1:
        # 单 qumode 操作，可能需要一些单 qumode 门
        return T_SINGLE
    else:
        # 多 qumode 操作，至少需要一个多 qumode 门
        # 根据复杂度估计可能需要多个
        complexity = estimate_complexity(expr)
        return complexity * T_MULTI

def estimate_complexity(expr: SymbolicOperator) -> int:
    """
    估计表达式的复杂度（需要多少个多 qumode 门）。
    这是一个简化的启发式估计。
    """
    if isinstance(expr, (BosonicOp, PauliOp)):
        return 1
    elif isinstance(expr, SumOp):
        # 和的复杂度 = 所有项的复杂度之和
        return sum(estimate_complexity(term) for term in expr.terms)
    elif isinstance(expr, ProductOp):
        # 乘积的复杂度 = 所有因子的复杂度之和
        return sum(estimate_complexity(factor) for factor in expr.factors if not isinstance(factor, Scalar))
    elif isinstance(expr, (CommutatorOp, AntiCommutatorOp)):
        # 对易子/反对易子通常需要分解
        return 2
    else:
        # 默认估计
        return 1

# --- 7. 表达式模式匹配和辅助函数 ---

def is_commutative(M: SymbolicOperator, N: SymbolicOperator) -> bool:
    """
    检查两个算符是否对易。
    简化实现：如果作用于不相交的 qumode 集合，则对易。
    """
    ## use numpy to check if the two operators are commutative
    if isinstance(M, ProductOp):
        for factor in M.factors:
            if not is_commutative(factor, N):
                return False
        return True
    elif isinstance(N, ProductOp):
        for factor in N.factors:
            if not is_commutative(M, factor):
                return False
        return True
    elif isinstance(M, SumOp):
        for term in M.terms:
            if not is_commutative(term, N):
                return False
        return True
    elif isinstance(N, SumOp):
        for term in N.terms:
            if not is_commutative(M, term):
                return False
        return True
    else:
        ## if M and N has different qumode then return true
    # If M and N act on different qumodes, then they commute
        def extract_qumodes(op):
            if hasattr(op, "qumodes"):
                return set(op.qumodes)
            elif isinstance(op, (ProductOp, SumOp)):
                # gather qumodes from all factors/terms
                result = set()
                elems = op.factors if isinstance(op, ProductOp) else op.terms
                for elem in elems:
                    result |= extract_qumodes(elem)
                return result
            else:
                return set()
        if extract_qumodes(M).isdisjoint(extract_qumodes(N)):
            return True
        # Fallback: by default assume they do not commute
        return not dagger(M) == N

def extract_scalar_factor(expr: SymbolicOperator) -> Tuple[complex, SymbolicOperator]:
    """
    从表达式中提取标量因子。
    返回 (标量值, 剩余表达式)
    """
    if isinstance(expr, ProductOp) and len(expr.factors) > 0:
        first_factor = expr.factors[0]
        if isinstance(first_factor, Scalar):
            remaining = ProductOp(expr.factors[1:]) if len(expr.factors) > 1 else Scalar(1.0)
            return (first_factor.value, remaining)
    return (1.0 + 0j, expr)

def create_B_operator(A: SymbolicOperator, alpha: complex = 1.0 + 0j) -> SymbolicOperator:
    """
    创建 B_{A} 算符的符号表示。
    B_{A} = exp(2i * alpha * (0 A; A† 0))
    """
    return BOp(A, alpha)

# --- 8. 分解规则实现 ---

def is_hermitian(op: SymbolicOperator) -> bool:
    """Check if operator is Hermitian (op == dagger(op))."""
    return op == dagger(op)

# Rule 1: exp(M t + N t) ≈ Trotter(M t, N t) -> (exp(M t / k) exp(N t / k))^k
def rule_1_trotter(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    scalar, inner = extract_scalar_factor(expr)
    if isinstance(inner, SumOp) and len(inner.terms) == 2:
        M_term, N_term = inner.terms
        # Assume scalar is t, but generalize
        k = 2  # Example k=2 for approximation
        result = []
        for _ in range(k):
            result.append(Hamiltonian(expr=ProductOp([Scalar(scalar / k), M_term])))
            result.append(Hamiltonian(expr=ProductOp([Scalar(scalar / k), N_term])))
        return result
    return []

# Rule 2: exp([M t, N t]) ≈ exp(M t) exp(N t) exp(-M t) exp(-N t)
def rule_2_bch(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    scalar, inner = extract_scalar_factor(expr)
    if isinstance(inner, CommutatorOp):
        M = inner.A
        N = inner.B
        # Assume scalar is t
        Mt = ProductOp([Scalar(scalar), M])
        Nt = ProductOp([Scalar(scalar), N])
        neg_Mt = ProductOp([Scalar(-scalar), M])
        neg_Nt = ProductOp([Scalar(-scalar), N])
        return [[
            Hamiltonian(expr=Mt),
            Hamiltonian(expr=Nt),
            Hamiltonian(expr=neg_Mt),
            Hamiltonian(expr=neg_Nt)
        ]]
    return []

# Rule 3: exp(t^2 [M, N]) -> exp([i t sigma_i N, i t sigma_i M]) if M, N Hermitian
def rule_3(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    scalar, inner = extract_scalar_factor(expr)
    if scalar.real > 0 and scalar.imag == 0 and isinstance(inner, CommutatorOp):
        M = inner.A
        N = inner.B
        if is_hermitian(M) and is_hermitian(N):
            t = scalar.real ** 0.5
            sigma_i = Y(0)  # Example, choose appropriate
            it_sigma_N = ProductOp([Scalar(1j * t), sigma_i, N])
            it_sigma_M = ProductOp([Scalar(1j * t), sigma_i, M])
            return [Hamiltonian(expr=CommutatorOp(it_sigma_N, it_sigma_M))]
    return []

# Rule 4: exp(-i t^2 sigma_i {M, N}) -> exp([i t sigma_j M, i t sigma_k N]) if M, N Hermitian
def rule_4(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    scalar, inner = extract_scalar_factor(expr)
    if scalar == -1j and isinstance(inner, ProductOp) and len(inner.factors) == 2:
        sigma_i, anticom = inner.factors
        if isinstance(sigma_i, PauliOp) and isinstance(anticom, AntiCommutatorOp):
            M = anticom.A
            N = anticom.B
            if is_hermitian(M) and is_hermitian(N):
                t = 1.0  # Extract t^2 from scalar if needed
                sigma_j = X(sigma_i.index)
                sigma_k = Y(sigma_i.index)
                it_sigma_j_M = ProductOp([Scalar(1j * t), sigma_j, M])
                it_sigma_k_N = ProductOp([Scalar(1j * t), sigma_k, N])
                return [Hamiltonian(expr=CommutatorOp(it_sigma_j_M, it_sigma_k_N))]
    return []

# Rule 5: exp(-i t^2 sigma_z [M, N]) -> exp([i t N, i t sigma_z M])
def rule_5(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    scalar, inner = extract_scalar_factor(expr)
    M = None
    N = None
    sigma_z = None
    conditions = {}
    if isinstance(inner, ProductOp) and len(inner.factors) == 2 and isinstance(inner.factors[0], PauliOp) and inner.factors[0].op_type == 'Z':
        sigma_z = inner.factors[0]
        comm = inner.factors[1]
        if isinstance(comm, CommutatorOp):
            M = comm.A
            N = comm.B
            conditions = {sigma_z.index: 0}
    elif isinstance(inner, CommutatorOp):
        M = inner.A
        N = inner.B
        qubit_idx = 0
        sigma_z = Z(qubit_idx)
        conditions = {qubit_idx: 0}
    if M is not None and N is not None and scalar == -1j:
        t = 1.0  # Adjust for t^2
        it_N = ProductOp([Scalar(1j * t), N])
        it_sigma_M = ProductOp([Scalar(1j * t), sigma_z, M])
        return [Hamiltonian(expr=CommutatorOp(it_N, it_sigma_M), conditions=conditions)]
    return []

# Rule 6: exp(t^2 sigma_z (M N - (M N)†)) -> exp([X i t B_N X, i t B_M]) if [M,N]=0
def rule_6(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    scalar, inner = extract_scalar_factor(expr)
    matched = False
    sigma_z = None
    conditions = {}
    if isinstance(inner, ProductOp) and len(inner.factors) == 2 and isinstance(inner.factors[0], PauliOp) and inner.factors[0].op_type == 'Z':
        sigma_z = inner.factors[0]
        diff = inner.factors[1]
        conditions = {sigma_z.index: 0}
        matched = True
    else:
        diff = inner
        qubit_idx = 0
        sigma_z = Z(qubit_idx)
        conditions = {qubit_idx: 0}
        matched = True
    if matched and isinstance(diff, SumOp) and len(diff.terms) == 2:
        term1, term2 = diff.terms
        if term2 == dagger(term1) * Scalar(-1):
            if isinstance(term1, ProductOp) and len(term1.factors) == 2:
                M, N = term1.factors
                if is_commutative(M, N):
                    t = scalar.real ** 0.5
                    B_M = create_B_operator(M, Scalar(1.0))
                    B_N = create_B_operator(N, Scalar(1.0))
                    it_B_M = ProductOp([Scalar(1j * t), B_M])
                    X_it_B_N_X = ProductOp([X(sigma_z.index), Scalar(1j * t), B_N, X(sigma_z.index)])
                    return [Hamiltonian(expr=CommutatorOp(X_it_B_N_X, it_B_M), conditions=conditions)]
    return []

# Rule 7: exp(i t^2 sigma_z (M N + (M N)†)) -> exp([S i t B_M S†, X i t B_N X]) if [M,N]=0
def rule_7(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    scalar, inner = extract_scalar_factor(expr)
    matched = False
    sigma_z = None
    conditions = {}
    if isinstance(inner, ProductOp) and len(inner.factors) == 2 and isinstance(inner.factors[0], PauliOp) and inner.factors[0].op_type == 'Z':
        sigma_z = inner.factors[0]
        summ = inner.factors[1]
        conditions = {sigma_z.index: 0}
        matched = True
    else:
        summ = inner
        qubit_idx = 0
        sigma_z = Z(qubit_idx)
        conditions = {qubit_idx: 0}
        matched = True
    if matched and isinstance(summ, SumOp) and len(summ.terms) == 2:
        term1, term2 = summ.terms
        if term2 == dagger(term1):
            if isinstance(term1, ProductOp) and len(term1.factors) > 2:
                ## split term1 into two parts
                len_factors = len(term1.factors)
                results = []
                for i in range(1, len_factors):
                    M = ProductOp(term1.factors[:i])
                    N = ProductOp(term1.factors[i:])
                    if is_commutative(M, N):

                        t = scalar.real ** 0.5
                        B_M = create_B_operator(M, Scalar(1.0))
                        B_N = create_B_operator(N, Scalar(1.0))
                        S = GateOp('S', sigma_z.index)  # Placeholder for S gate
                        S_dag = dagger(S)
                        S_it_B_M_Sdag = ProductOp([S, Scalar(1j * t), B_M, S_dag])
                        X_it_B_N_X = ProductOp([X(sigma_z.index), Scalar(1j * t), B_N, X(sigma_z.index)])
                        results.append([Hamiltonian(expr=CommutatorOp(S_it_B_M_Sdag, X_it_B_N_X), conditions=conditions)])
                return results
    return []

# Rule 8: exp(-2 i t (MN 0 ; 0 -MN)) -> exp(-i t sigma_z [M,N] - i t sigma_z {M,N}) if M,N Hermitian
def rule_8(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    scalar, inner = extract_scalar_factor(expr)
    if scalar == -2j and isinstance(inner, SumOp) and len(inner.terms) == 2:
        term1, term2 = inner.terms
        if isinstance(term1, ProductOp) and isinstance(term2, ProductOp) and term2 == term1 * Scalar(-1):  # Simulate diag
            MN = term1
            if isinstance(MN, ProductOp) and len(MN.factors) == 2:
                M, N = MN.factors
                if is_hermitian(M) and is_hermitian(N):
                    t = 1.0  # Assume t in scalar
                    qubit_idx = 0
                    sigma_z = Z(qubit_idx)
                    comm = CommutatorOp(M, N)
                    anticom = AntiCommutatorOp(M, N)
                    part1 = ProductOp([Scalar(-1j * t), sigma_z, comm])
                    part2 = ProductOp([Scalar(-1j * t), sigma_z, anticom])
                    return [Hamiltonian(expr=SumOp([part1, part2]), conditions={qubit_idx: 0})]
    return []

# Rule 9: exp(2 i t^2 (MN 0 ; 0 -MN)) -> exp([S i t B_M S†, X i t B_N X]) if [M,N]=0 and MN = (MN)†
def rule_9(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    scalar, inner = extract_scalar_factor(expr)
    if scalar == 2j and isinstance(inner, SumOp) and len(inner.terms) == 2:
        term1, term2 = inner.terms
        if isinstance(term1, ProductOp) and term2 == term1 * Scalar(-1):
            MN = term1
            if MN == dagger(MN) and isinstance(MN, ProductOp) and len(MN.factors) == 2:
                M, N = MN.factors
                if is_commutative(M, N):
                    t = 1.0
                    B_M = create_B_operator(M, Scalar(1.0))
                    B_N = create_B_operator(N, Scalar(1.0))
                    qubit_idx = 0
                    S = Z(qubit_idx)
                    S_dag = dagger(S)
                    S_it_B_M_Sdag = ProductOp([S, Scalar(1j * t), B_M, S_dag])
                    X_it_B_N_X = ProductOp([X(qubit_idx), Scalar(1j * t), B_N, X(qubit_idx)])
                    return [Hamiltonian(expr=CommutatorOp(S_it_B_M_Sdag, X_it_B_N_X))]
    return []

# Rule 10: exp(2 i t B_MN) -> X exp(t sigma_y (MN - (MN)†) + i t sigma_x (MN + (MN)†)) X if [M,N]=0
def rule_10(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    scalar, inner = extract_scalar_factor(expr)
    if isinstance(inner, ProductOp):  # Assume inner is B_MN
        # Extract M, N from B
        if len(inner.factors) == 3 and isinstance(inner.factors[1], BOp):
            gate, B_op, gate_dag = inner.factors
            A = B_op.A
            if isinstance(A, ProductOp):
                ## Split A into two parts
                len_factors = len(A.factors)
                results = []

                for i in range(1, len_factors):
                    M = ProductOp(A.factors[:i])
                    N = ProductOp(A.factors[i:])
                    if is_commutative(M, N):
                        t = 1.0
                        qubit_idx = 0
                        X_gate = X(qubit_idx)
                        MN = ProductOp([M, N])
                        diff = SumOp([MN, dagger(MN) * Scalar(-1)])
                        summ = SumOp([MN, dagger(MN)])
                        part1 = ProductOp([Scalar(t), Y(qubit_idx), diff])
                        part2 = ProductOp([Scalar(1j * t), X(qubit_idx), summ])
                        middle = SumOp([part1, part2])
                        if gate.op_type == 'X':
                            results.append([Hamiltonian(expr=middle)])
                        else:
                            results.append([gate, X_gate, Hamiltonian(expr=middle), X_gate, gate_dag])
                return results  
    return []

# Rule 11: exp(i t (2 MN 0 ; 0 -N M - (N M)†)) -> exp([S i t B_M S†, X i t B_N X]) if MN = (MN)†
def rule_11(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    scalar, inner = extract_scalar_factor(expr)
    if scalar == 1j and isinstance(inner, SumOp) and len(inner.terms) == 2:
        term1, term2 = inner.terms
        if isinstance(term1, ProductOp) and term1.factors[0] == Scalar(2):
            MN = term1.factors[1]
            if isinstance(term2, SumOp) and term2.terms[0] == MN * Scalar(-1) and term2.terms[1] == dagger(MN) * Scalar(-1):
                if MN == dagger(MN):
                    # Extract M, N assuming MN = M * N, but N M = N * M if commutative
                    if isinstance(MN, ProductOp) and len(MN.factors) == 2:
                        M, N = MN.factors
                        t = 1.0
                        B_M = create_B_operator(M, Scalar(1.0))
                        B_N = create_B_operator(N, Scalar(1.0))
                        qubit_idx = 0
                        S = Z(qubit_idx)
                        S_dag = dagger(S)
                        S_it_B_M_Sdag = ProductOp([S, Scalar(1j * t), B_M, S_dag])
                        X_it_B_N_X = ProductOp([X(qubit_idx), Scalar(1j * t), B_N, X(qubit_idx)])
                        return [Hamiltonian(expr=CommutatorOp(S_it_B_M_Sdag, X_it_B_N_X))]
    return []

# Rule 12: B_a = exp(2 i alpha (0 a ; a† 0)) -> sequence if alpha = alpha*
def rule_12(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    scalar, inner = extract_scalar_factor(expr)
    if scalar == 2j and isinstance(inner, SumOp) and len(inner.terms) == 2:
        term1, term2 = inner.terms
        if term1 == a(0) and term2 == a_dag(0):  # Simplify index 0
            alpha = 1.0  # Extract from scalar if needed
            if alpha == alpha.conjugate():
                phase1 = ProductOp([Scalar(1j * (math.pi / 2)), a_dag(0), a(0)])
                disp_y = ProductOp([Scalar(1j * alpha), SumOp([a_dag(0), a(0)]), Y(0)])
                phase2 = ProductOp([Scalar(-1j * (math.pi / 2)), a_dag(0), a(0)])
                disp_x = ProductOp([Scalar(1j * alpha), SumOp([a_dag(0), a(0)]), X(0)])
                return [
                    Hamiltonian(expr=phase1),
                    Hamiltonian(expr=disp_y),
                    Hamiltonian(expr=phase2),
                    Hamiltonian(expr=disp_x)
                ]
    return []

# Rule 13: Similar to 12 but for a†
def rule_13(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    scalar, inner = extract_scalar_factor(expr)
    if scalar == 2j and isinstance(inner, SumOp) and len(inner.terms) == 2:
        term1, term2 = inner.terms
        if term1 == a_dag(0) and term2 == a(0):
            alpha = 1.0
            if alpha == alpha.conjugate():
                phase1 = ProductOp([Scalar(1j * (math.pi / 2)), a_dag(0), a(0)])
                disp_y = ProductOp([Scalar(1j * alpha), SumOp([a_dag(0), a(0)]), Y(0)])
                phase2 = ProductOp([Scalar(-1j * (math.pi / 2)), a_dag(0), a(0)])
                disp_x = ProductOp([Scalar(-1j * alpha), SumOp([a_dag(0), a(0)]), X(0)])
                return [
                    Hamiltonian(expr=phase1),
                    Hamiltonian(expr=disp_y),
                    Hamiltonian(expr=phase2),
                    Hamiltonian(expr=disp_x)
                ]
    return []

# Rule 14: exp( (P1 P2 ... Pn) (alpha a_k† - alpha* a_k) ) -> multi-qubit controlled displacement
def rule_14(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    if isinstance(expr, ProductOp) and len(expr.factors) == 2:
        paulis, disp = expr.factors
        if isinstance(paulis, ProductOp) and all(isinstance(f, PauliOp) for f in paulis.factors):
            if isinstance(disp, SumOp) and len(disp.terms) == 2:
                term1, term2 = disp.terms
                if isinstance(term1, ProductOp) and isinstance(term1.factors[0], Scalar) and isinstance(term1.factors[1], BosonicOp) and term1.factors[1].is_creation:
                    alpha = term1.factors[0].value
                    # Assume term2 = - conj(alpha) * a_k, check if matches
                    conj_alpha = alpha.conjugate()
                    if isinstance(term2, ProductOp) and term2.factors[0] == Scalar(-conj_alpha) and isinstance(term2.factors[1], BosonicOp) and not term2.factors[1].is_creation and term2.factors[1].index == term1.factors[1].index:
                        # Decomposition is complex; return placeholder for RHS of Eq (11)
                        return [Hamiltonian(expr=expr, is_leaf=True)]  # Mark as leaf if native
    return []

# Rule 15: exp(2 i alpha^2 P1 P2 ... Pn) -> multi-Pauli exponential
def rule_15(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    scalar, inner = extract_scalar_factor(expr)
    if scalar == 2j and isinstance(inner, ProductOp) and all(isinstance(f, PauliOp) for f in inner.factors):
        # Decomposition to RHS of Eq (9); assume it's leaf or specific sequence
        return [Hamiltonian(expr=inner, is_leaf=True)]
    return []

# Rule 16: All native gates in Table 2 -> corresponding base gates
def rule_16(H: Hamiltonian) -> List[Hamiltonian]:
    expr = H.expr
    # Example matching for CBS: exp(i theta (a b† + a† b))
    if isinstance(expr, SumOp) and len(expr.terms) == 2:
        term1, term2 = expr.terms
        if isinstance(term1, ProductOp) and len(term1.factors) == 2 and isinstance(term1.factors[0], BosonicOp) and not term1.factors[0].is_creation and isinstance(term1.factors[1], BosonicOp) and term1.factors[1].is_creation:
            if term2 == dagger(term1):
                return [Hamiltonian(expr=expr, is_leaf=True, cost=T_MULTI)]
    # Add more patterns for other native gates...
    return []

# 转换定理：sigma_z 转换
def sigma_z_conversion(H: Hamiltonian) -> List[Hamiltonian]:
    """
    转换定理：通过 H * sigma_z * H = sigma_x 和 
    S^dagger * H * sigma_z * H * S = sigma_y
    将 sigma_y 和 sigma_x 变成 sigma_z
    
    输入: exp(alpha * sigma_x * M) 或 exp(alpha * sigma_y * M)
    输出: 转换为 exp(alpha * sigma_z * M') 的形式
    """
    expr = H.expr
    
    # 模式匹配：检查是否包含 sigma_x 或 sigma_y
    # 简化处理：如果表达式包含 X 或 Y Pauli 算符，尝试转换
    
    # 提取 qumode
    qumodes = list(H.qumodes)
    if qumodes:
        qumode_idx = qumodes[0]
    else:
        qumode_idx = 0
    
    # 创建转换门序列
    # H * sigma_z * H = sigma_x -> 需要 H 门
    # S^dagger * H * sigma_z * H * S = sigma_y -> 需要 S^dagger, H, S
    
    # 简化：假设可以识别并转换
    # 这里返回转换后的形式
    return [Hamiltonian(expr=expr, is_leaf=False)]

# --- 9. 规则调度器 ---

# 规则查找表：根据表达式类型决定可应用的规则
def get_applicable_rules(H: Hamiltonian) -> List[Callable[[Hamiltonian], List[Hamiltonian]]]:
    """
    根据哈密顿量的表达式类型，返回可应用的分解规则列表。
    """
    rules = []
    expr = H.expr
    
    # Rule 1: Trotter（适用于 SumOp）
    if isinstance(expr, SumOp):
        if len(expr.terms) == 2:
            H_1, H_2 = expr.terms
            ## check if H_1 and H_2 are Hermitian
            if is_hermitian(H_1) and is_hermitian(H_2):
                rules.append(rule_1_trotter)
            elif H_1 == dagger(H_2):
                ## check if H_1 and H_2 are basic gates
                
                if len(H_1.factors) <= 2:
                    ## H_1 +H_2 is a basic gate
                    rules.append(rule_16)
                    return rules
                else:
                    ## split H_1 and H_2 into two parts
                    rules.append(rule_7)
            elif H_1 == - dagger(H_2):
                if len(H_1.factors) > 2:
                    rules.append(rule_6)
        elif len(expr.terms) % 2 == 1:
            all_hermitian = True
            for term in expr.terms:
                if not is_hermitian(term):
                    all_hermitian = False
                    break
            if all_hermitian:
                rules.append(rule_1_trotter)
        
    
    # Rule 2: BCH（适用于 CommutatorOp）
    if isinstance(expr, CommutatorOp):
        rules.append(rule_2_bch)
    
    # # Rule 3: 对易子分解（适用于 CommutatorOp）
    # if isinstance(expr, CommutatorOp):
    #     rules.append(rule_3)
    
    # # Rule 4: 反对易子分解（适用于 AntiCommutatorOp）
    # if isinstance(expr, AntiCommutatorOp):
    #     rules.append(rule_4)
    
    # # Rule 5: sigma_z 对易子（需要模式匹配）
    # # 简化：如果表达式包含对易子，尝试应用
    # if isinstance(expr, CommutatorOp):
    #     rules.append(rule_5)
    
    # Rule 8, 9: 矩阵形式（需要模式匹配）
    # 简化实现
    
    # Rule 10: B 算符（需要模式匹配）
    if isinstance(expr, ProductOp):
        contain_B = False
        for factor in expr.factors:
            if isinstance(factor, BOp):
                contain_B = True
                break
        if contain_B:
            rules.append(rule_10)
    # Rule 11: 矩阵形式
    # 简化实现
    
    # Rule 12, 13: B_a 和 B_{a^dagger}（需要模式匹配）
    # 简化实现
    
    # Rule 14, 15: 多 qubit 控制和多 Pauli（需要模式匹配）
    # 简化实现
    
    # Rule 16: 原生门转换（适用于简单表达式）
    # if len(H.qumodes) <= 2:
    #     rules.append(rule_16)
    
    # sigma_z 转换（适用于包含 Pauli 算符的表达式）
    # 检查是否包含 X 或 Y Pauli 算符
    if isinstance(expr, ProductOp) and len(expr.factors) == 2 and isinstance(expr.factors[0], PauliOp) and isinstance(expr.factors[1], PauliOp) and expr.factors[0].op_type in ['X', 'Y'] and expr.factors[1].op_type in ['X', 'Y']:
        rules.append(sigma_z_conversion)
    
    return rules

def contains_pauli(expr: SymbolicOperator, pauli_types: List[str]) -> bool:
    """
    检查表达式是否包含指定类型的 Pauli 算符。
    """
    if isinstance(expr, PauliOp):
        return expr.op_type in pauli_types
    elif isinstance(expr, SumOp):
        return any(contains_pauli(term, pauli_types) for term in expr.terms)
    elif isinstance(expr, ProductOp):
        return any(contains_pauli(factor, pauli_types) for factor in expr.factors)
    elif isinstance(expr, CommutatorOp):
        return contains_pauli(expr.A, pauli_types) or contains_pauli(expr.B, pauli_types)
    elif isinstance(expr, AntiCommutatorOp):
        return contains_pauli(expr.A, pauli_types) or contains_pauli(expr.B, pauli_types)
    return False

# --- 10. 核心 A* 搜索 ---

def get_neighbors(node: SearchNode) -> List[SearchNode]:
    """
    A* 的核心 "expand" 函数。
    找到第一个可分解的 H，应用所有适用规则，并生成新的子节点。
    """
    neighbors: List[SearchNode] = []
   
    # 1. 找到第一个要分解的 H
    H_to_expand: Optional[Hamiltonian] = None
    target_block_index: int = -1   
    # 遍历所有块和块中的所有 H
    for i, block in enumerate(node.blocks):
        for H in block:
            if not H.is_leaf:
                H_to_expand = H
                target_block_index = i
                break
        if H_to_expand:
            break
            
    # 如果没有找到（即所有都是叶节点），则返回空列表
    if not H_to_expand:
        return []

    # 2. 将 blocks 展平回一个 H 列表
    flat_H_list: List[Hamiltonian] = []
    H_index_in_flat_list = -1
    
    current_index = 0
    for i, block in enumerate(node.blocks):
        for H in block:
            if H == H_to_expand:
                H_index_in_flat_list = current_index
            flat_H_list.append(H)
            current_index += 1

    # 3. 应用所有适用规则
    applicable_rules = get_applicable_rules(H_to_expand)
    
    for rule_func in applicable_rules:
        # 4. 生成新的 H 列表
        new_H_fragments = rule_func(H_to_expand)
        for new_H_fragment in new_H_fragments:
            if not new_H_fragment:
                continue
            new_blocks = compact(new_H_fragment)
            new_g_cost = calculate_g_cost(new_blocks)
            new_node = SearchNode(
                g_cost=new_g_cost,
                blocks=new_blocks,
                parent=node,
                rule_applied=f"{rule_func.__name__} on {H_to_expand}"
            )
            neighbors.append(new_node)
    return neighbors

def find_optimal_path(root_H: Hamiltonian) -> Optional[SearchNode]:
    """A* 搜索算法主循环。"""
    
    # 根节点是包含 root_H 的单个块
    root_blocks = (frozenset[Hamiltonian]({root_H}),)
    root_g_cost = calculate_g_cost(root_blocks)
    root_node = SearchNode(g_cost=root_g_cost, blocks=root_blocks)

    open_set = [root_node]  # 优先队列 (min-heap)
    visited = {root_node.blocks}  # 存储已访问的 *状态* (blocks)

    iteration = 0
    
    while open_set:
        iteration += 1
        
        # 1. 获取 f_cost 最低的节点
        current_node = heapq.heappop(open_set)
        
        # 2. 检查是否为目标
        # 我们的目标是 h_cost = 0 (所有 H 都是叶节点)
        if current_node.calculate_h_cost() == 0:
            print(f"\n--- 搜索成功！在 {iteration} 次迭代后找到最优路径 ---")
            return current_node
            
        # 3. 扩展邻居
        for neighbor in get_neighbors(current_node):
            
            # 检查是否已经访问过这个 *状态*
            if neighbor.blocks not in visited:
                
                # 添加到 open set 和 visited
                heapq.heappush(open_set, neighbor)
                visited.add(neighbor.blocks)
                
        if iteration > 5000:  # 防止无限循环
            print("搜索超时！")
            return None

    print("未找到解。")
    return None

def reconstruct_path(node: SearchNode):
    """回溯并打印最优路径。"""
    path = []
    current = node
    while current:
        path.append(current)
        current = current.parent
    
    print(f"\n--- 最终代价 (时间): {path[0].g_cost} ns ---")
    print("\n--- 分解路径 (逆序) ---")
    for i, step in enumerate(reversed(path)):
        print(f"\n步骤 {i}: {step.rule_applied}")
        print(f"  g_cost (so far): {step.g_cost} ns")
        print(f"  h_cost (est.): {step.calculate_h_cost()} ns")
        print("  状态 (并行块):")
        for block in step.blocks:
            print(f"    {block}")


