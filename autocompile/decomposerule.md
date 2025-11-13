\begin{tabular}{|l|l|l|l|l|l|}
\hline Rules & Operator Template & Conditions & Decomposition Output & Reference & Precision \\
\hline 1 & $\exp (M t+N t) \approx \operatorname{Trotter}(M t, N t)$ & & $(\exp (M t / k) \exp (N t / k))^k$ & Trotterization & Approx \\
\hline 2 & $\exp ([M t, N t]) \approx \mathrm{BCH}(M t, N t)$ & & $\exp (M t) \exp (N t) \exp (-M t) \exp (-N t)$ & BCH & Approx \\
\hline 3 & $\exp \left(t^2[M, N]\right)$ & M, $N$ Hermitian & $\exp \left(\left[i t \sigma_i N, i t \sigma_i M\right]\right)$ & [20] & Exact \\
\hline 4 & $\exp \left(-i t^2 \sigma_i\{M, N\}\right)$ & M, $N$ Hermitian & $\exp \left(\left[i t \sigma_j M, i t \sigma_k N\right]\right)$ & [20] & Exact \\
\hline 5 & $\exp \left(-i t^2 \sigma_z[M, N]\right)$ & & $\exp \left(\left[\left(i t N, i t \sigma_z M\right]\right)\right.$ & This paper & Exact \\
\hline 6 & $\exp \left(t^2 \sigma_z\left(\left(M N-(M N)^{\dagger}\right)\right)\right)$ & $[M, N]=0$ & $\exp \left(\left[X \cdot i t \mathcal{B}_N \cdot X, i t \mathcal{B}_M\right]\right)$ & [20] & Exact \\
\hline 7 & $\exp \left(i t^2 \sigma_z\left(\left(M N+(M N)^{\dagger}\right)\right)\right)$ & $[M, N]=0$ & $\exp \left(\left[S \cdot i t \mathcal{B}_M \cdot S^{\dagger}, X \cdot i t \mathcal{B}_N \cdot X\right]\right)$ & [20] & Exact \\
\hline 8 & $\exp \left(-2 i t\left(\begin{array}{cc}M N & 0 \\ 0 & -M N\end{array}\right)\right)$ & M, $N$ Hermitian & $\exp \left(-i t \sigma_z[M, N]-i t \sigma_z\{M, N\}\right)$ & This paper & Exact \\
\hline 9 & $\exp \left(2 i t^2\left(\begin{array}{cc}M N & 0 \\ 0 & -M N\end{array}\right)\right)$ & $$
\begin{aligned}
& {[M, N]=0} \\
& M N=(M N)^{\dagger}
\end{aligned}
$$ & $\exp \left(\left[\left(S \cdot i t \mathcal{B}_M \cdot S^{\dagger}, X \cdot i t \mathcal{B}_N \cdot X\right]\right)\right.$ & [20] & Exact \\
\hline 10 & $\exp \left(2 i t \mathcal{B}_{M N}\right)$ & $[M, N]=0$ & $X \cdot \exp \left(t \sigma_y\left(M N-(M N)^{\dagger}\right)+i t \sigma_x\left(M N+(M N)^{\dagger}\right)\right) \cdot X$ & [20] & Exact \\
\hline 11 & $\exp \left(i t\left(\begin{array}{cc}2 M N & 0 \\ 0 & -N M-(N M)^{\dagger}\end{array}\right)\right)$ & $M N=(M N)^{\dagger}$ & $\exp \left(\left[S \cdot i t \mathcal{B}_M \cdot S^{\dagger}, X \cdot i t \mathcal{B}_N \cdot X\right]\right)$ & [20] & Exact \\
\hline 12 & $\mathcal{B}_a=\exp \left(2 i \alpha\left(\begin{array}{cc}0 & a \\ a^{\dagger} & 0\end{array}\right)\right)$ & $\alpha=\alpha^*$ & $\exp \left(i(\pi / 2) a^{\dagger} a\right) \exp \left(i\left(\alpha\left(a^{\dagger}+a\right)\right) \otimes \sigma_y\right) \exp \left(-i(\pi / 2) a^{\dagger} a\right) \exp \left(i\left(\alpha\left(a^{\dagger}+a\right)\right) \otimes \sigma_x\right)$ & [20] & Approx \\
\hline 13 & $\mathcal{B}_{a^{\dagger}}=\exp \left(2 i \alpha\left(\begin{array}{cc}0 & a^{\dagger} \\ a & 0\end{array}\right)\right)$ & $\alpha=\alpha^*$ & $\exp \left(i(\pi / 2) a^{\dagger} a\right) \exp \left(i\left(\alpha\left(a^{\dagger}+a\right)\right) \otimes \sigma_y\right) \exp \left(-i(\pi / 2) a^{\dagger} a\right) \exp \left(-i\left(\alpha\left(a^{\dagger}+a\right)\right) \otimes \sigma_x\right)$ & This paper & Approx \\
\hline 14 & $e^{\left(P_1 P_2 \cdots P_n\right)\left(\alpha a_k^{\dagger}-\alpha^2 a_k\right)}$ & & Multi-qubit-controlled displacement: Right hand side (RHS) of Equation (11) first line & [28] & Exact \\
\hline 15 & $e^{2 i \alpha^2 P_1 P_2 \cdots P_n}$ & & Multi-Pauli Exponential: Right hand side (RHS) of Equation (9) first line & This Paper & Exact \\
\hline 16 & All Native Gates RHS in Table 2 & & All Native Gates Left Hand Side (LHS) Table 2 & [28] & Exact \\
\hline
\end{tabular}