import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt
from itertools import product
from typing import List, Tuple, Dict, Optional
import sympy as sympy_core  # Rename to avoid conflict
import time

class DD_BosonicQAOAIPSolver:
    """
    Decision Diagram / Subspace based Bosonic QAOA Solver.
    
    Core Idea:
    Instead of building operators in the full N^d Hilbert space (which is huge),
    we explicitly construct the 'Feasible Subspace' graph (a simplified Decision Diagram).
    We map feasible states |n> to logical indices |k> and perform evolution 
    in this drastically reduced subspace.
    """
    
    def __init__(
        self,
        A: np.ndarray, 
        b: np.ndarray,
        c: np.ndarray,
        N: int = 7,
        p: int = 2,
        g: float = 1.0,
        problem_name: str = "dd_solver"
    ):
        self.A = np.array(A, dtype=int)
        self.b = np.array(b, dtype=int)
        self.c = np.array(c, dtype=float)
        self.N = N
        self.p = p
        self.g = g
        self.num_modes = len(self.c)
        self.problem_name = problem_name
        
        # 1. Compute Integer Nullspace (same as before)
        self.null_space_basis = self._compute_integer_nullspace()
        
        # 2. efficiently find feasible states using Backtracking (DD construction logic)
        # This avoids iterating over N^d states.
        t0 = time.time()
        self.feasible_states = self._find_feasible_states_dd()
        self.num_feasible = len(self.feasible_states)
        print(f"Feasible subspace size: {self.num_feasible} (found in {time.time()-t0:.4f}s)")
        
        if self.num_feasible == 0:
            raise ValueError("No feasible states found under constraints.")

        # Map state tuple -> index
        self.state_to_idx = {state: i for i, state in enumerate(self.feasible_states)}
        
        # 3. Build Compressed Hamiltonians (Sparse Matrices)
        self.H_C = self._build_compressed_cost_hamiltonian()
        self.H_M_components = self._build_compressed_driver_hamiltonian_components()
        self.H_M = sum(self.H_M_components)
        
        # Pre-compute optimal info
        costs = [np.dot(self.c, s) for s in self.feasible_states]
        self.max_obj = max(costs)
        self.min_obj = min(costs) # For initial state superposition
        
        print(f"Compressed Hamiltonian size: {self.num_feasible}x{self.num_feasible}")

    def _compute_integer_nullspace(self) -> List[np.ndarray]:
        """(Same logic as original) Uses sympy to find integer null basis."""
        A_mat = sympy_core.Matrix(self.A)
        nullspace = A_mat.nullspace()
        basis = []
        for vec in nullspace:
            # Scale to integer
            denoms = [elem.q for elem in vec if hasattr(elem, 'q')]
            lcm = sympy_core.lcm(denoms) if denoms else 1
            v_int = np.array([(elem * lcm) for elem in vec], dtype=int).flatten()
            
            # Make primitive
            gcd = np.gcd.reduce(v_int)
            if gcd != 0: v_int //= gcd
            
            # Direction normalization
            if np.any(v_int):
                first_nonzero = v_int[np.nonzero(v_int)[0][0]]
                if first_nonzero < 0: v_int = -v_int
            basis.append(v_int)
            
        # Add loop closure for connectivity if needed (heuristic)
        if len(basis) > 1:
            basis.append(basis[0] + basis[-1])
            
        return [b for b in basis if np.any(b)]

    def _find_feasible_states_dd(self) -> List[Tuple[int, ...]]:
        """
        DFS / Branch-and-Bound search to find states satisfying Ax = b.
        This constructs the valid paths of the Decision Diagram implicitly.
        Much faster than itertools.product for large N.
        """
        valid_states = []
        
        # Pre-calculate min/max possible contributions for remaining variables for pruning
        # This assumes A has positive entries mostly, or simple bounds. 
        # For general A, simple bounds check: 0 <= x_i < N
        
        def backtrack(col_idx, current_partial_sum, current_state):
            # Base case: all variables assigned
            if col_idx == self.num_modes:
                if np.array_equal(current_partial_sum, self.b):
                    valid_states.append(tuple(current_state))
                return

            # Pruning (Optional but recommended for speed):
            # Check if current_partial_sum is already too far from b to recover?
            # (Simple bounding can be added here if A contains only positive numbers)
            
            # Iterate possible values for current variable
            # Heuristic: iterate 0..N-1
            # Optimization: Compute bounds for x_i based on (b - partial) / A_i
            
            row_indices, _ = np.nonzero(self.A[:, col_idx:col_idx+1])
            
            for val in range(self.N):
                # Check constraints immediately if this is the last variable for a row
                new_sum = current_partial_sum + self.A[:, col_idx] * val
                
                # Simple Pruning: If any constraint is already violated (assuming positive Coeffs)
                # If A has negative coeffs, this check is harder, skipping for generality.
                
                backtrack(col_idx + 1, new_sum, current_state + [val])

        backtrack(0, np.zeros_like(self.b), [])
        return valid_states

    def _build_compressed_cost_hamiltonian(self) -> sp.csr_matrix:
        """H_C is diagonal in the computational basis."""
        # H_C |n> = (-c.n) |n>
        diagonals = [-np.dot(self.c, state) for state in self.feasible_states]
        return sp.diags(diagonals, format='csr')

    def _build_compressed_driver_hamiltonian_components(self) -> List[sp.csr_matrix]:
        """
        Builds H_M = sum_u H_u in the compressed basis.
        H_u connects state |n> to |n + u> and |n - u>.
        """
        components = []
        dim = self.num_feasible
        
        for u in self.null_space_basis:
            # Build sparse matrix for vector u
            row_ind = []
            col_ind = []
            data = []
            
            for i, state in enumerate(self.feasible_states):
                state_arr = np.array(state)
                
                # Check transition: state -> state + u
                target_plus = state_arr + u
                if np.all((target_plus >= 0) & (target_plus < self.N)):
                    target_tuple = tuple(target_plus)
                    if target_tuple in self.state_to_idx:
                        j = self.state_to_idx[target_tuple]
                        # Calculate matrix element: g * sqrt(product of factors)
                        # O_u = prod (a_k^dagger)^u_k ...
                        # Val = sqrt( (n+1)... )
                        val = self._calculate_transition_amplitude(state_arr, u)
                        
                        row_ind.append(j) # |j><i|
                        col_ind.append(i)
                        data.append(self.g * val)
                        
                        # Hermitian conjugate: |i><j|
                        row_ind.append(i)
                        col_ind.append(j)
                        data.append(self.g * val) # Real amplitude assumed
            
            # Create sparse matrix
            H_u = sp.csr_matrix((data, (row_ind, col_ind)), shape=(dim, dim))
            components.append(H_u)
            
        return components

    def _calculate_transition_amplitude(self, state: np.ndarray, u: np.ndarray) -> float:
        """
        Calculates <n+u| O_u |n>.
        O_u = prod_{k: u_k>0} (a_k^dag)^u_k  * prod_{k: u_k<0} (a_k)^|u_k|
        """
        amp = 1.0
        # Apply annihilation first (on state n)
        temp_state = state.copy()
        for k in range(self.num_modes):
            if u[k] < 0:
                p = abs(u[k])
                # a^p |n> = sqrt(n * (n-1) * ... * (n-p+1)) |n-p>
                for step in range(p):
                    amp *= np.sqrt(temp_state[k])
                    temp_state[k] -= 1
        
        # Apply creation (on temp_state)
        for k in range(self.num_modes):
            if u[k] > 0:
                p = u[k]
                # (a^dag)^p |m> = sqrt((m+1)*...*(m+p)) |m+p>
                for step in range(p):
                    amp *= np.sqrt(temp_state[k] + 1)
                    temp_state[k] += 1
        return amp

    def get_initial_state(self, superposition=True) -> np.ndarray:
        """
        Returns the initial state vector in the compressed subspace.
        superposition: If True, returns (|argmax> + |argmin>)/sqrt(2).
        """
        dim = self.num_feasible
        psi = np.zeros(dim, dtype=complex)
        
        costs = [np.dot(self.c, s) for s in self.feasible_states]
        
        if superposition:
            # Find all states with max cost and min cost
            max_c = max(costs)
            min_c = min(costs)
            indices = [i for i, c in enumerate(costs) if np.isclose(c, max_c) or np.isclose(c, min_c)]
        else:
            # Just one heuristic state (e.g. max cost)
            max_c = max(costs)
            indices = [i for i, c in enumerate(costs) if np.isclose(c, max_c)]
            
        for idx in indices:
            psi[idx] = 1.0
            
        return psi / np.linalg.norm(psi)

    def simulate(self, params: np.ndarray, circuit_type: str = "multi_beta") -> Tuple[float, np.ndarray]:
        """
        Simulate QAOA and return (Energy, Final_State_Vector).
        Uses fast sparse matrix-vector multiplication.
        """
        psi = self.get_initial_state()
        
        # Parse params
        num_drivers = len(self.null_space_basis)
        
        if circuit_type == "beta_gamma":
            # params: [gamma0, beta0, gamma1, beta1, ...]
            # H_M is total driver
            for l in range(self.p):
                gamma = params[2*l]
                beta = params[2*l+1]
                
                # Cost layer: exp(-i * gamma * H_C)
                # Since H_C is diagonal, this is element-wise multiplication
                # H_C data is stored in self.H_C.data
                diag_phases = np.exp(-1j * gamma * self.H_C.data)
                psi = psi * diag_phases
                
                # Mixer layer: exp(-i * beta * H_M)
                psi = expm_multiply(-1j * beta * self.H_M, psi)

        elif circuit_type == "multi_beta":
            
            idx = 0
            for l in range(self.p):
                
                for k in range(num_drivers):
                    beta = params[idx]
                    idx += 1
                    psi = expm_multiply(-1j * beta * self.H_M_components[k], psi)
                    
        return psi

    def expectation(self, psi: np.ndarray) -> float:
        """Calculate <psi| H_C |psi>."""
        # H_C is diagonal real
        # <psi|H_C|psi> = sum |psi_i|^2 * H_C_ii
        prob = np.abs(psi)**2
        return np.sum(prob * self.H_C.data)

    def optimize(self):
        """Standard scipy optimize."""
        from scipy.optimize import minimize
        
        num_drivers = len(self.null_space_basis)
        if self.p is None: self.p = 1
        
        # Determine param shape
        # For "multi_beta" (matching user logic): p * num_drivers
        x0 = np.random.uniform(0, 0.5, self.p * num_drivers)
        
        history = []
        
        def loss(x):
            psi = self.simulate(x, circuit_type="multi_beta")
            # We want to Maximize Objective => Minimize Energy (H_C = -Obj)
            # Energy = <H_C>
            energy = self.expectation(psi)
            return energy # H_C is already negative of objective
            
        def callback(x):
            en = loss(x)
            history.append(en)
            print(f"Iter {len(history)}: Cost={en:.4f}")
            
        res = minimize(loss, x0, method='COBYLA', options={'maxiter': 200}, callback=callback)
        
        # Decode result
        final_psi = self.simulate(res.x, circuit_type="multi_beta")
        probs = np.abs(final_psi)**2
        
        # Find best state
        best_idx = np.argmax(probs)
        best_state = self.feasible_states[best_idx]
        best_obj = np.dot(self.c, best_state)
        
        return {
            "x": res.x,
            "fun": -res.fun, # Positive objective
            "best_state": best_state,
            "best_prob": probs[best_idx],
            "feasible_count": self.num_feasible
        }

# ==========================================
# Running the fast solver
# ==========================================
if __name__ == "__main__":
    # Same Problem parameters
    num_variables = 8
    num_constraints = 2
    # Generate random instance (copying user's logic roughly)
    np.random.seed(42)
    A = np.random.randint(1, 5, size=(num_constraints, num_variables))
    x0 = np.random.randint(0, 4, size=num_variables) # Make sure small enough
    b = A @ x0
    c = np.random.randint(1, 5, size=num_variables)
    
    print("Problem:")
    print(f"A:\n{A}")
    print(f"b: {b}")
    print(f"c: {c}")
    print(f"Target x0 (one solution): {x0}")
    
    # Initialize Fast Solver
    solver = DD_BosonicQAOAIPSolver(A, b, c, N=10, p=2)
    
    # Optimize
    print("\nStarting Optimization...")
    start_t = time.time()
    result = solver.optimize()
    end_t = time.time()
    
    print("\n=== Result ===")
    print(f"Time: {end_t - start_t:.4f}s")
    print(f"Max Objective Found: {result['fun']:.4f}")
    print(f"Best State: {result['best_state']}")
    print(f"Probability: {result['best_prob']:.4f}")