# TreeCompiler - 量子计算树编译器

基于 A* 算法的哈密顿量分解优化系统，用于将量子哈密顿量分解为最优的基础门序列。

## 功能特性

- 基于 `SymbolicOperator` 的符号算符系统
- A* 搜索算法优化分解路径
- 支持并行块压缩，最大化并行执行
- 实现所有分解规则（Rule 1-16）
- sigma_z 转换定理
- 基于执行时间的代价优化（单 qumode 门: 20ns, 多 qumode 门: 700ns）

## 安装

### 使用 uv 安装

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装项目
uv pip install -e .

# 或安装开发依赖
uv pip install -e ".[dev]"
```

### 使用 pip 安装

```bash
pip install -e .
```

## 使用方法

```python
from operator import a, a_dag
from main import Hamiltonian, find_optimal_path, reconstruct_path

# 创建哈密顿量
H_expr = a(1) * a_dag(2) + a_dag(2) * a_dag(1)
root_H = Hamiltonian(expr=H_expr, is_leaf=False)

# 运行 A* 搜索
optimal_node = find_optimal_path(root_H)

# 打印结果
if optimal_node:
    reconstruct_path(optimal_node)
```

## 项目结构

- `operator.py`: 符号算符系统（SymbolicOperator）
- `main.py`: A* 搜索算法和分解规则实现
- `decomposerule.md`: 分解规则文档
- `分解流程.md`: 分解流程示例

## 开发

```bash
# 安装开发依赖
uv pip install -e ".[dev]"

# 运行代码格式化
black .
ruff check .

# 运行类型检查
mypy .
```

## 许可证

[待添加]

