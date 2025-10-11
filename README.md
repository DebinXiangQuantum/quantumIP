## Quantum solver for mixed-integer programming
### Integer constraints encoding
cvIP/integerCons.py contains the code to implement the encoding of constraints with a driver Hamiltonian.
cvIP/qaoa.py is the demo of integer programming using the above driver Hamiltonian.

#### realizing constraints encoding

##### Core Idea: Translating Subspace Confinement to Qumodes

The magic of **Choco-Q** lies not in filtering but in **subspace confinement**.
The algorithm constructs a driver Hamiltonian ( H_d ) that cannot escape the subspace of valid solutions.

To extend this to **qumodes**, we must translate three key components:

1. Variables (binary → integer)
2. Constraint operator ( \hat{C} )
3. Commuting driver Hamiltonian ( H_d )

---

##### Step 1: The Qumode Constraint Operator ( \hat{C} )

This is the most straightforward translation.

In the qumode framework (e.g., Khosravi *et al.*), non-negative integer variables ( n_i ) are represented by the eigenstates of the photon number operator

$$
\hat{n}_i = \hat{a}_i^\dagger \hat{a}_i
$$

For a set of linear equality constraints:

$$
\sum_i c_i n_i = c
$$

the corresponding **quantum constraint operator** is:

$$
\hat{C} = \sum_i c_i \hat{n}_i
$$

A quantum state ( |\psi\rangle ) satisfies the constraint if

$$
\hat{C} |\psi\rangle = c |\psi\rangle
$$

Thus, the constraint subspace is spanned by all Fock basis states
( |n_1, n_2, \dots, n_k\rangle ) satisfying the classical equation.

---

##### Step 2: Constructing the Commuting Qumode Driver ( H_d )

This is the heart of the challenge.

We need a driver ( H_d ) that:

$$
[H_d, \hat{C}] = 0 \quad \text{but} \quad [H_d, H_o] \neq 0
$$

A naive choice $ H_d = f(\hat{n}_1, \hat{n}_2, \dots) $ would commute with both $ \hat{C} $ **and** $ H_o $, violating the second condition. Therefore, $ H_d $ must involve **creation** $( \hat{a}^\dagger )$ and **annihilation** $( \hat{a} )$ operators, which do *not* commute with $ \hat{n}_i $.

---

### Example: Total Photon Number Conservation

Consider the constraint:

$$
n_1 + n_2 = K
$$

so that

$$
\hat{C} = \hat{n}_1 + \hat{n}_2
$$

We seek a driver that preserves total photon number.
The **beam-splitter Hamiltonian** does precisely this:

$$
H_{\text{BS}} = g (\hat{a}_1^\dagger \hat{a}_2 + \hat{a}_2^\dagger \hat{a}_1)
$$

It’s well known that:

$$
H_{\text{BS}}, \hat{n}_1 + \hat{n}_2 = 0
$$

Hence, $ H_d = H_{\text{BS}} $ acts as a valid driver: it coherently mixes valid configurations like $ |K, 0\rangle $, $ |K-1, 1\rangle $, etc., while staying within the constraint subspace.


### General Case: The Null Space Method for Qumodes

For a general constraint:

$$
\hat{C} = \sum_i c_i \hat{n}_i
$$

we can generalize the **null-space approach** from Choco-Q:

1. **Find the Null Space**
   Compute integer vectors $ {u^{(1)}, u^{(2)}, \dots} $ satisfying
   $$
   C \cdot u = 0 \quad \Rightarrow \quad \sum_i c_i u_i = 0
   $$

2. **Translate Vectors to Operators**
   Each null-space vector $ u $ corresponds to an operator that *shifts photon numbers* by the amounts $ u_i $:

   $$
   \hat{O}_u =
   \left(\prod_{j | u_j > 0} (\hat{a}_j^\dagger)^{u_j}\right)
   \left(\prod_{k | u_k < 0} (\hat{a}_k)^{|u_k|}\right)
   $$

   The corresponding driver term is:

   $$
   H_u = g  (\hat{O}_u + \hat{O}_u^\dagger)
   $$

   Since $ \sum_i c_i u_i = 0 $, we have:

   $$
   [H_u, \hat{C}] = 0
   $$

3. **Combine All Drivers**

   The full driver is the sum over all basis vectors of the constraint null space:

   $$
   H_d = \sum_j H_{u^{(j)}}
   $$

This construction is the **qumode analogue** of the Choco-Q driver.


## Step 3: Verifying the QAOA Condition $ [H_d, H_o] \neq 0 $

The condition holds naturally:

* The **objective Hamiltonian** is typically:

  $$
  H_o = \sum_j d_j \hat{n}*j + \sum*{ij} Q_{ij} \hat{n}_i \hat{n}_j
  $$

* The **driver Hamiltonian** uses ( \hat{a} ) and ( \hat{a}^\dagger ):

  $$
  [\hat{n}, \hat{a}] = -\hat{a}, \quad [\hat{n}, \hat{a}^\dagger] = \hat{a}^\dagger
  $$

Thus:

$$
[H_d, H_o] \neq 0
$$
[todo](): How to compile the Unitary into native gates.

### Use Supperposition can reduce the prob gap
[todo](): How to prepare a supperposition state ?


### Real constraints encoding
cvDriverHamiltonian.py is the implementation of encoding real constraints with the driver Hamiltonian, which uses the  position to represent the real number variable.

continuous variables using $\hat{x}_i$ for variable $x_i$, define the constraint operator as $\hat{C} = \sum_{i=1}^n c_i \hat{x}_i$. The goal is subspace confinement where $\hat{C} |\psi\rangle = c |\psi\rangle$, but in continuous-variable quantum computing (CVQC), this "subspace" is a hyperplane in the infinite-dimensional position space, with measure zero. Strict hard constraints are theoretically ideal but practically approximate due to non-normalizable $\hat{x}$ eigenstates.

Analogous to Choco-Q's null-space method (and its qumode-integer extension in the provided discussion), construct $H_d$ as follows:

1. **Identify the Null Space**: For constraint matrix $\mathbf{C}$ (here a row vector $(c_1, \dots, c_n)$), find an orthonormal basis $\{ \vec{u}^j \}_{j=1}^{n-1}$ for the null space, where $\mathbf{C} \cdot \vec{u}^j = 0$ (i.e., $\sum_i c_i u_i^j = 0$).

2. **Construct Driver Terms**: The driver must generate displacements along null directions without altering $\hat{C}$. In CVQC, translations in position space are generated by momentum operators $\hat{p}_i = \frac{1}{i\sqrt{2}} (\hat{a}_i - \hat{a}_i^\dagger)$, with $[\hat{x}_i, \hat{p}_k] = i \delta_{ik}$.

   For each basis vector $\vec{u}^j$, define $$H_{u^j} = g \sum_{k=1}^n u_k^j \hat{p}_k$$, where $g$ is a coupling strength. Verify commutation:
   $$
   [H_{u^j}, \hat{C}] = \sum_{i,k} c_i u_k^j [\hat{p}_k, \hat{x}_i] = -i \sum_k c_k u_k^j = -i (\mathbf{C} \cdot \vec{u}^j) = 0.
   $$
   The full driver is $H_d = \sum_j H_{u^j}$.

3. **QAOA Formulation**: The QAOA ansatz alternates $e^{-i\beta H_d}$ and $e^{-i\gamma H_o}$, where $H_o$ is the objective encoded as a polynomial in $\{\hat{x}_i\}$ (e.g., $H_o = \sum_i d_i \hat{x}_i + \sum_{ij} Q_{ij} \hat{x}_i \hat{x}_j$). Non-commutation $[H_d, H_o] \neq 0$ holds since $H_o$ involves $\hat{x}_i$ and $H_d$ involves $\hat{p}_k$, with $[\hat{x}_i, \hat{p}_k] \neq 0$ for $i=k$.

4. **Initial State Preparation**: Start with a Gaussian state (e.g., squeezed vacuum) where $\langle \hat{C} \rangle = c$ and variance $\text{Var}(\hat{C})$ is minimized. Evolution preserves $\langle \hat{C} \rangle = c$, but $\text{Var}(\hat{C})$ persists, yielding approximate hard constraints. For tighter enforcement, prepare highly squeezed states along the $\hat{C}$ direction.


### Mixed Constraints in Quantum Solvers for MIP

In the context of bosonic or continuous-variable quantum computing for mixed-integer programming (MIP), constraints can indeed mix integer variables (encoded via photon-number operators $\hat{n}_i = \hat{a}_i^\dagger \hat{a}_i$ on qumodes) and real variables (encoded via position operators $\hat{x}_j$). The challenge is to construct a driver Hamiltonian $H_d$ that enforces **subspace confinement** for such a mixed constraint operator $\hat{C}$, satisfying $[H_d, \hat{C}] = 0$ while ensuring $[H_d, H_o] \neq 0$ for the objective Hamiltonian $H_o$. Below, I formalize the construction, building on the null-space method from the integer (qumode) and real (CV) cases discussed in the provided materials.

#### Formal Setup
Consider a single linear equality constraint mixing $m$ integer variables $\{n_k\}_{k=1}^m \in \mathbb{Z}_{\geq 0}$ and $\ell$ real variables $\{x_r\}_{r=1}^\ell \in \mathbb{R}$:

$$
\sum_{k=1}^m c_k n_k + \sum_{r=1}^\ell d_r x_r = c,
$$

where $\mathbf{c} = (c_1, \dots, c_m) \in \mathbb{R}^m$, $\mathbf{d} = (d_1, \dots, d_\ell) \in \mathbb{R}^\ell$, and $c \in \mathbb{R}$. Assume distinct modes for integers (qumodes) and reals (CV modes), so operators on different modes commute: $[\hat{n}_k, \hat{x}_r] = 0$ for all $k, r$.

The quantum constraint operator is

$$
\hat{C} = \sum_{k=1}^m c_k \hat{n}_k + \sum_{r=1}^\ell d_r \hat{x}_r.
$$

The feasible subspace $\mathcal{S}_c$ consists of (approximate) eigenstates $|\psi\rangle$ with $\hat{C} |\psi\rangle = c |\psi\rangle$. For integers, this is discrete (Fock states); for reals, it is continuous (position eigenstates, though non-normalizable, leading to approximations via Gaussians).

The objective $H_o$ is typically a quadratic form, e.g.,

$$
H_o = \sum_k e_k \hat{n}_k + \sum_r f_r \hat{x}_r + \sum_{k,k'} Q_{kk'} \hat{n}_k \hat{n}_{k'} + \sum_{r,r'} R_{rr'} \hat{x}_r \hat{x}_{r'} + \text{cross terms},
$$

with cross terms like $\sum_{k,r} S_{kr} \hat{n}_k \hat{x}_r$.

#### Construction of the Mixed Driver Hamiltonian $H_d$
The null-space method extends naturally to the hybrid (discrete-continuous) setting. The classical constraint defines a codimension-1 affine hyperplane in the $( \mathbf{n}, \mathbf{x} ) \in \mathbb{Z}_{\geq 0}^m \times \mathbb{R}^\ell$ space. To mix states within $\mathcal{S}_c$, we identify **null directions** $\mathbf{u}^{(j)} = (\mathbf{u}_{\text{int}}^{(j)}, \mathbf{u}_{\text{real}}^{(j)}) \in \mathbb{Z}^m \times \mathbb{R}^\ell$ (for $j = 1, \dots, p$) satisfying

$$
\mathbf{c} \cdot \mathbf{u}_{\text{int}}^{(j)} + \mathbf{d} \cdot \mathbf{u}_{\text{real}}^{(j)} = 0,
$$

where $\mathbf{u}_{\text{int}}^{(j)}$ has integer components (to preserve discreteness of $n_k$) and $\mathbf{u}_{\text{real}}^{(j)}$ has real components. These form a generating set for the null space (not necessarily orthonormal, as the space is hybrid). For example, choose a basis where each $\mathbf{u}_{\text{int}}^{(j)}$ is a standard basis vector in a reduced integer space, compensated by $\mathbf{u}_{\text{real}}^{(j)} = - (\mathbf{d}^\top)^{-1} \mathbf{c} \cdot e_j$ (assuming $\mathbf{d}$ invertible; otherwise, project).

For each null direction $\mathbf{u}^{(j)}$, construct a **shift operator** $\hat{O}_{u^{(j)}}$ that displaces integers by $\mathbf{u}_{\text{int}}^{(j)}$ and reals by $\mathbf{u}_{\text{real}}^{(j)}$, preserving the constraint value:

- Integer shift (discrete, via creation/annihilation): For positive $u_{k}^{(j)} > 0$, apply $(\hat{a}_k^\dagger)^{u_k^{(j)}}$; for negative $u_{k}^{(j)} < 0$, apply $(\hat{a}_k)^{|u_k^{(j)}|}$. Thus,

  $$
  \hat{O}_{\text{int}}^{(j)} = \prod_{k: u_k^{(j)} > 0} \left( \hat{a}_k^\dagger \right)^{u_k^{(j)}} \prod_{k: u_k^{(j)} < 0} \left( \hat{a}_k \right)^{|u_k^{(j)}|}.
  $$

- Real shift (continuous, via momentum $\hat{p}_r = i \sqrt{\frac{\hbar}{2}} (\hat{a}_r^\dagger - \hat{a}_r)$, with $[\hat{x}_r, \hat{p}_s] = i \hbar \delta_{rs}$): The displacement by $\delta_r = u_r^{(j)}$ is generated by $\exp\left( -i \frac{\delta_r}{\hbar} \hat{p}_r \right)$, so

  $$
  \hat{O}_{\text{real}}^{(j)} = \exp\left( -i \sum_r \frac{u_r^{(j)}}{\hbar} \hat{p}_r \right).
  $$

The full shift operator is the tensor product (since modes are distinct):

$$
\hat{O}_{u^{(j)}} = \hat{O}_{\text{int}}^{(j)} \otimes \hat{O}_{\text{real}}^{(j)}.
$$

The Hermitian driver term is

$$
H_{u^{(j)}} = g_j \left( \hat{O}_{u^{(j)}} + \hat{O}_{u^{(j)}}^\dagger \right),
$$

where $g_j > 0$ is a tunable coupling. The total driver is the sum over a basis of null directions:

$$
H_d = \sum_{j=1}^p H_{u^{(j)}}.
$$

#### Verification of Subspace Confinement: $[H_d, \hat{C}] = 0$
Consider a state $|\psi\rangle \in \mathcal{S}_c$ with $\hat{C} |\psi\rangle = c |\psi\rangle$. Applying $\hat{O}_{u^{(j)}}$ shifts the expectation values: $\Delta \langle \hat{n}_k \rangle = u_k^{(j)}$ (integer) and $\Delta \langle \hat{x}_r \rangle = u_r^{(j)}$ (real). The change in $\langle \hat{C} \rangle$ is

$$
\Delta \langle \hat{C} \rangle = \sum_k c_k u_k^{(j)} + \sum_r d_r u_r^{(j)} = 0,
$$

by null-space construction. Since $\hat{n}_k$ and $\hat{x}_r$ act on distinct modes (commuting across types), and shifts are exact within each subspace, $\hat{O}_{u^{(j)}} |\psi\rangle \in \mathcal{S}_c$. Similarly for $\hat{O}_{u^{(j)}}^\dagger$. Thus, $H_{u^{(j)}} |\psi\rangle \in \mathcal{S}_c$, implying

$$
[H_d, \hat{C}] |\psi\rangle = 0 \quad \forall |\psi\rangle \in \mathcal{S}_c.
$$

For the full space, the commutator vanishes on $\mathcal{S}_c$ (the relevant subspace), achieving confinement.

#### Non-Commutation with Objective: $[H_d, H_o] \neq 0$
The objective $H_o$ generally depends on $\{\hat{n}_k, \hat{x}_r\}$. Since $H_d$ involves $\hat{a}_k, \hat{a}_k^\dagger$ (for integers) and $\hat{p}_r \sim \hat{a}_r^\dagger - \hat{a}_r$ (for reals), we have canonical commutation relations:

$$
[\hat{n}_k, \hat{a}_k] = -\hat{a}_k, \quad [\hat{n}_k, \hat{a}_k^\dagger] = \hat{a}_k^\dagger, \quad [\hat{x}_r, \hat{p}_r] = i \hbar.
$$

Cross-mode terms commute, but intra-mode non-commutation ensures $[H_d, H_o] \neq 0$ unless $H_o$ is invariant under null shifts (rare, e.g., if $H_o$ is constant on $\mathcal{S}_c$). For QAOA, this enables variational optimization via alternating $e^{-i \beta H_d}$ and $e^{-i \gamma H_o}$.

#### Simple Example: One Integer, One Real
Constraint: $n_1 + x_2 = c$. Null direction: $\mathbf{u}^{(1)} = (-1, 1)$. Then,

$$
\hat{O}_{u^{(1)}} = \hat{a}_1 \exp\left( -i \frac{1}{\hbar} \hat{p}_2 \right), \quad H_d = g \left( \hat{O}_{u^{(1)}} + \hat{O}_{u^{(1)}}^\dagger \right).
$$

Applying to $|n_1, x_2 = c - n_1\rangle$ yields $|n_1 - 1, x_2 + 1\rangle$, preserving the constraint. Commutation holds as shown; non-commutation with, e.g., $H_o = \hat{n}_1 \hat{x}_2$ follows from $[\hat{a}_1, \hat{n}_1] \neq 0$.

#### Practical Considerations and Approximations
- **Basis Selection**: Compute null directions numerically (e.g., via SVD on the augmented constraint matrix). For multiple constraints, sum drivers over independent null spaces.
- **Compilation to Native Gates**: Trotterize $e^{-i \beta H_d}$ into beam-splitter-like gates for integers (from $\hat{a}^\dagger \hat{a}'$) and displacement/squeezing for reals. Cross terms require controlled displacements.
- **Initial State**: Use a product state with $\langle \hat{n}_k \rangle$ near feasible integers and a squeezed Gaussian for reals minimizing $\text{Var}(\hat{C})$.
- **Scalability**: Discrete shifts limit integer range; approximate large shifts via QAOA layers. For reals, finite squeezing bounds precision.
- **Relation to Separate Constraints**: As noted, if constraints decouple (C1 integer-only, C2 real-only), $H_d = H_{d1} + H_{d2}$ works since $[H_{d1}, \hat{C}_2] = 0$. Mixed constraints couple modes, but the construction above handles this.

This hybrid driver enables exact (in principle) confinement for mixed constraints, extending Choco-Q to full MIP on bosonic hardware. Implementation in `cvIP/mixedCons.py` could follow the qumode/real templates, with null-space computation via SymPy/NumPy.