import abc
from dataclasses import dataclass
from typing import List, Union, Tuple
from zlib import DEF_BUF_SIZE

# --- 1. 抽象基类 (Abstract Base Class) ---
# 
# 所有的算符，无论是基础的还是组合的，都继承自这个类。
# 它定义了所有算符共有的接口 (加、减、乘、对易等)。

class SymbolicOperator(abc.ABC):
    """
    哈密顿量符号系统的抽象基类。
    """

    @abc.abstractmethod
    def __repr__(self) -> str:
        """返回该算符的符号化字符串表示。"""
        pass

    # --- 运算符重载 (Operator Overloading) ---
    # 
    # 这些方法允许我们使用 Python 的原生运算符 (如 + * -) 
    # 来“组合”我们的符号算符，自动创建 AST 节点。

    def __add__(self, other):
        # H1 + H2
        other = _ensure_op(other)
        return SumOp([self, other])

    def __radd__(self, other):
        # 5 + H1
        other = _ensure_op(other)
        return SumOp([other, self])

    def __sub__(self, other):
        # H1 - H2  ->  H1 + (-1 * H2)
        other = _ensure_op(other)
        return SumOp([self, ProductOp([Scalar(-1.0), other])])

    def __rsub__(self, other):
        # 5 - H1  ->  5 + (-1 * H1)
        other = _ensure_op(other)
        return SumOp([other, ProductOp([Scalar(-1.0), self])])

    def __mul__(self, other):
        # H1 * H2
        other = _ensure_op(other)
        return ProductOp([self, other])

    def __rmul__(self, other):
        # 5 * H1
        other = _ensure_op(other)
        return ProductOp([other, self])
        
    def __neg__(self):
        # -H1 -> -1 * H1
        return ProductOp([Scalar(-1.0), self])

    # --- 对易/反对易方法 (Commutator Methods) ---

    def commutator(self, other: 'SymbolicOperator') -> 'CommutatorOp':
        """计算 [self, other]"""
        return CommutatorOp(self, _ensure_op(other))

    def anti_commutator(self, other: 'SymbolicOperator') -> 'AntiCommutatorOp':
        """计算 {self, other}"""
        return AntiCommutatorOp(self, _ensure_op(other))

def _ensure_op(val: Union['SymbolicOperator', complex, int, float]) -> 'SymbolicOperator':
    """
    一个辅助函数，确保参与运算的项都是 SymbolicOperator。
    如果输入是数字，它会将其包装成 Scalar 算符。
    """
    if isinstance(val, SymbolicOperator):
        return val
    if isinstance(val, (int, float, complex)):
        return Scalar(complex(val))
    return NotImplemented


# --- 2. 叶节点 (Leaf Nodes) ---
# 
# 代表构建哈密顿量的基础“积木”。
# @dataclass(frozen=True) 使得这些对象是不可变的 (immutable)，
# 这在符号计算中是一个很好的实践，可以防止意外修改。

@dataclass(frozen=True)
class Scalar(SymbolicOperator):
    """
    代表一个标量 (系数)。
    例如: 5.0, (3+2j)
    """
    value: complex

    def __repr__(self) -> str:
        return f"{self.value}"

@dataclass(frozen=True)
class PauliOp(SymbolicOperator):
    """
    代表一个作用在单个 qubit 上的 Pauli 算符。
    例如: X_0, Z_3
    """
    op_type: str # 'I', 'X', 'Y', 'Z'
    index: int

    def __repr__(self) -> str:
        if self.op_type == 'I':
            return f"I_{self.index}"
        return f"{self.op_type}_{self.index}"

    def __pow__(self, exponent: int) -> Union['PauliOp', 'ProductOp', 'Scalar']:
        """支持幂运算，如 X(0)**2 返回 I (identity)"""
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Exponent must be a non-negative integer")

        if exponent == 0:
            # P^0 = 1 (identity)
            return Scalar(1.0)
        elif exponent == 1:
            # P^1 = P
            return self
        elif exponent % 2 == 0:
            # For Pauli operators: X^2 = Y^2 = Z^2 = I, and I^even = I
            return PauliOp('I', self.index)
        else:
            # For odd exponents > 1, we need to multiply
            return ProductOp([self] * exponent)


@dataclass(frozen=True)
class GateOp(SymbolicOperator):
    """
    代表一个量子门算符。
    例如: S, S_dag, X, Y, Z
    """
    op_type: str # 'S', 'S_dag', "Hadamard"
    index: int
    def __repr__(self) -> str:
        return f"{self.op_type}_{self.index}"
    def __pow__(self, exponent: int) -> 'GateOp':
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Exponent must be a non-negative integer")
        if exponent == 0:
            return Scalar(1.0)
        elif exponent == 1:
            return self
        else:
            return ProductOp([self] * exponent)
    def dagger(self) -> 'SymbolicOperator':
        if self.op_type == 'S':
            return GateOp('S_dag', self.index)
        elif self.op_type == 'S_dag':
            return GateOp('S', self.index)
        elif self.op_type == 'Hadamard':
            return GateOp('Hadamard', self.index)
        else:
            raise ValueError(f"Unsupported gate type: {self.op_type}")

@dataclass(frozen=True)
class BosonicOp(SymbolicOperator):
    """
    代表一个作用在单个 qumode 上的玻色子算符 (a 或 a_dag)。
    例如: a_1, a_dag_0
    """
    is_creation: bool # True 表示 a_dag, False 表示 a
    index: int

    def __repr__(self) -> str:
        op_name = "a_dag" if self.is_creation else "a"
        return f"{op_name}_{self.index}"

    def __pow__(self, exponent: int) -> 'ProductOp':
        """支持幂运算，如 a_dag(2)**3 返回 a_dag(2) * a_dag(2) * a_dag(2)"""
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Exponent must be a non-negative integer")

        if exponent == 0:
            # a^0 = 1 (identity)
            return Scalar(1.0)
        elif exponent == 1:
            # a^1 = a
            return self
        else:
            # a^n = a * a * ... * a (n times)
            return ProductOp([self] * exponent)


# --- 3. 内部节点 (Internal Nodes) ---
# 
# 代表各种“运算规则”，它们将其他算符组合起来。
#
# 注意：SumOp 和 ProductOp 在构造时实现了“扁平化” (flattening) 逻辑。
# 例如：SumOp(SumOp(A, B), C) 会被自动简化为 SumOp(A, B, C)。
# 这使得 AST 结构更清晰、更规范。

class SumOp(SymbolicOperator):
    """
    代表多个算符的加和。
    例如: H1 + H2 + H3
    """
    def __init__(self, terms: List[SymbolicOperator]):
        flat_terms = []
        for t in terms:
            if isinstance(t, SumOp):
                # 扁平化: (A + B) + C  ->  A + B + C
                flat_terms.extend(t.terms)
            else:
                flat_terms.append(t)
        # 使用 tuple 确保不可变性
        self.terms: Tuple[SymbolicOperator, ...] = tuple(flat_terms)

    def __repr__(self) -> str:
        return f"({' + '.join(map(str, self.terms))})"
        
    # --- 为不可变性和字典使用提供支持 ---
    def __hash__(self):
        return hash(self.terms)
    def __eq__(self, other):
        return isinstance(other, SumOp) and self.terms == other.terms

## substract operation


class ProductOp(SymbolicOperator):
    """
    代表多个算符的乘积 (顺序敏感)。
    例如: H1 * H2 * H3
    """
    def __init__(self, factors: List[SymbolicOperator]):
        flat_factors = []
        total_scalar = 1.0 + 0j
        
        for f in factors:
            if isinstance(f, ProductOp):
                # 扁平化: (A * B) * C  ->  A * B * C
                flat_factors.extend(f.factors)
            elif isinstance(f, Scalar):
                # 合并标量: (c1 * A) * c2 -> (c1*c2) * A
                total_scalar *= f.value
            else:
                flat_factors.append(f)

        # 将合并后的标量放在最前面
        if total_scalar == 1.0 and len(flat_factors) > 0:
             self.factors: Tuple[SymbolicOperator, ...] = tuple(flat_factors)
        else:
             self.factors: Tuple[SymbolicOperator, ...] = (Scalar(total_scalar), *flat_factors)

    def __repr__(self) -> str:
        return f"({' * '.join(map(str, self.factors))})"

    # --- 为不可变性和字典使用提供支持 ---
    def __hash__(self):
        return hash(self.factors)
    def __eq__(self, other):
        return isinstance(other, ProductOp) and self.factors == other.factors

@dataclass(frozen=True)
class CommutatorOp(SymbolicOperator):
    """
    代表对易子: [A, B] = A*B - B*A
    """
    A: SymbolicOperator
    B: SymbolicOperator

    def __repr__(self) -> str:
        return f"[{self.A}, {self.B}]"

@dataclass(frozen=True)
class AntiCommutatorOp(SymbolicOperator):
    """
    代表反对易子: {A, B} = A*B + B*A
    """
    A: SymbolicOperator
    B: SymbolicOperator

    def __repr__(self) -> str:
        return f"{{{self.A}, {self.B}}}"

@dataclass(frozen=True)
class BOp(SymbolicOperator):
    """
    代表 B_{A} 算符: exp(2i * alpha * (0 A; A† 0))
    """
    A: SymbolicOperator
    alpha: complex = (1 + 0j)
    def __repr__(self) -> str:
        return f"B_{{{self.A}}}({self.alpha})"
    
    def __pow__(self, exponent: int) -> 'BOp':
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Exponent must be a non-negative integer")
        if exponent == 0:
            return Scalar(1.0)
        elif exponent == 1:
            return self

# --- 辅助函数 (Helper Functions) ---
def X(i: int) -> PauliOp:
    return PauliOp('X', i)

def Y(i: int) -> PauliOp:
    return PauliOp('Y', i)

def Z(i: int) -> PauliOp:
    return PauliOp('Z', i)



def I(i: int) -> PauliOp:
    return PauliOp('I', i)

def a(i: int) -> BosonicOp:
    return BosonicOp(is_creation=False, index=i)

def a_dag(i: int) -> BosonicOp:
    return BosonicOp(is_creation=True, index=i)

# --- 辅助函数：提取 qumode 集合 ---
def extract_qumodes(op: SymbolicOperator) -> set:
    """
    从符号算符中提取所有涉及的 qumode 索引集合。
    用于判断两个算符是否对易（作用于不同的 qumode 集合时对易）。
    """
    qumodes = set()
    
    if isinstance(op, BosonicOp):
        qumodes.add(op.index)
    elif isinstance(op, PauliOp):
        # Pauli 算符作用于 qubit，但在这个系统中我们可能需要将其映射到 qumode
        # 这里假设 Pauli 的 index 就是 qumode index
        qumodes.add(op.index)
    elif isinstance(op, Scalar):
        # 标量不涉及任何 qumode
        pass
    elif isinstance(op, SumOp):
        for term in op.terms:
            qumodes.update(extract_qumodes(term))
    elif isinstance(op, ProductOp):
        for factor in op.factors:
            qumodes.update(extract_qumodes(factor))
    elif isinstance(op, CommutatorOp):
        qumodes.update(extract_qumodes(op.A))
        qumodes.update(extract_qumodes(op.B))
    elif isinstance(op, AntiCommutatorOp):
        qumodes.update(extract_qumodes(op.A))
        qumodes.update(extract_qumodes(op.B))
    
    return qumodes

# --- 示例代码（已注释，避免导入时执行） ---
# if __name__ == "__main__":
#     # --- 示例 1: 构建简单的哈密顿量 ---
#     print("--- 示例 1 ---")
#     # H = X_0 * Z_1 + 0.5 * Y_0
#     H1 = X(0) * Z(1)
#     H2 = 0.5 * Y(0)
#     H = H1 + H2
#     
#     print(f"H = {H}")
#     # 输出: H = ((1.0, X_0, Z_1) + (0.5, Y_0))
#     # 注意: ProductOp 自动将系数 1.0 和 0.5 提取到了前面
#     
#     # --- 示例 2: 包含玻色子算符 (Rabi 模型) ---
#     print("\n--- 示例 2 ---")
#     omega_c = 1.0
#     omega_q = 2.0
#     g = 0.1
#     
#     # H = omega_c * a_dag(0) * a(0) + 0.5 * omega_q * Z(0) + g * X(0) * (a(0) + a_dag(0))
#     H_cavity = omega_c * a_dag(0) * a(0)
#     H_qubit = 0.5 * omega_q * Z(0)
#     H_int = g * X(0) * (a(0) + a_dag(0))
#     
#     H_rabi = H_cavity + H_qubit + H_int
#     
#     print(f"H_rabi = {H_rabi}")
#     # 输出: H_rabi = (((1.0, a_dag_0, a_0) + (1.0, Z_0)) + (0.1, X_0, (a_0 + a_dag_0)))
#     
#     # --- 示例 3: 包含对易子 (符合您的定义) ---
#     print("\n--- 示例 3 ---")
#     # H = [X_0, Y_1] - 3.0 * {Z_0, X_1}
#     term1 = X(0).commutator(Y(1))
#     term2 = Z(0).anti_commutator(X(1))
#     H_comm = term1 - 3.0 * term2
#     
#     print(f"H_comm = {H_comm}")
#     # 输出: H_comm = ([X_0, Y_1] + (-3.0, {Z_0, X_1}))