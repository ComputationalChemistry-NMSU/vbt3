"""
How a Hubbard U reshapes a covalent bond: the H2 example.

The basis at s=0, zero_ii=True is the 4-determinant Sz=0 space:
    |aA|   double occupation on orbital a  (ionic)
    |aB|   alpha on a, beta on b           (covalent)
    |bA|   alpha on b, beta on a           (covalent)
    |bB|   double occupation on orbital b  (ionic)

The standard valence-bond pictures are linear combinations:
    |cov>  = (|aB| + |bA|) / sqrt(2)       Heitler-London singlet
    |ion>  = (|aA| + |bB|) / sqrt(2)       symmetric ionic combination

The A_1 block of H (on the two symmetric singlets) is

        | cov          ion
    ---------------------------
    cov |  0           2 h
    ion |  2 h          U

with eigenvalues  E = U/2 +/- sqrt((U/2)^2 + 4 h^2).

The ground-state energy and the covalent/ionic weights can be written
in closed form in terms of U and t = |h|.  Both are derived end-to-end
here by the symbolic VBT machinery, then compared to the closed form.
"""
import os
import sys
import numpy as np
import sympy as sp
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from vbt3 import Molecule, FixedPsi
from vbt3.fixed_psi import generate_dets


# ------------------------------------------------------------------------
# 1. Build the 4x4 H, S, H2 symbolically via vbt3
# ------------------------------------------------------------------------
m = Molecule(
    zero_ii=True,
    interacting_orbs=['ab'],
    subst={'h': ('H_ab',), 's': ('S_ab',)},
    subst_2e={'U': ('1111',)},
    max_2e_centers=1,                      # on-site U only
)
P = generate_dets(1, 1, 2)
print("Basis:", [p.dets[0].det_string for p in P])

H_1e = m.build_matrix(P, op='H')
S    = m.build_matrix(P, op='S')
H_2e = m.o2_matrix(P)
H    = H_1e + H_2e

print("\nS:"); sp.pprint(sp.simplify(S))
print("\nH (1e + 2e):"); sp.pprint(sp.simplify(H))


# ------------------------------------------------------------------------
# 2. Closed form for the A_1 block (s = 0)
# ------------------------------------------------------------------------
h, s, U_sym = sp.symbols('h s U')
H_s0 = H.subs(s, 0)
print("\nH at s = 0:"); sp.pprint(H_s0)

# Symmetric VB basis:  |cov>, |ion>
T = sp.Matrix([[0, 1, 1, 0],
               [1, 0, 0, 1]]) / sp.sqrt(2)     # rows = cov, ion ; cols = dets

H_A1 = T * H_s0 * T.T
print("\nH in the {cov, ion} basis:")
sp.pprint(sp.simplify(H_A1))

E_sym = sp.Symbol('E')
chi = H_A1.charpoly(E_sym).as_expr()
roots = sp.solve(chi, E_sym)
roots_s = [sp.simplify(r) for r in roots]
print("\nEigenvalues of the A_1 block:")
for r in roots_s:
    print(f"  {r}")

E_gs = [r for r in roots_s if sp.simplify(r - roots_s[0]) == 0
        and sp.limit(r, U_sym, 0).subs(h, -1) == -2][0] if False else roots_s[1]
# Pick the branch with - sign (ground state for h < 0)
E_gs = (U_sym - sp.sqrt(U_sym**2 + 16*h**2)) / 2
print(f"\nGround state (exact closed form):  E(U, h) = {E_gs}")
print(f"  -> E(U, h=-1) = {sp.simplify(E_gs.subs(h, -1))}")


# ------------------------------------------------------------------------
# 3. Covalent vs ionic weights as a function of U (closed form)
# ------------------------------------------------------------------------
# Ground-state eigenvector in the A_1 block:
#   (H_A1 - E_gs I) c = 0.  Solve for ratio c_ion / c_cov.
c_cov, c_ion = sp.symbols('c_cov c_ion', real=True)
eq1 = (H_A1[0, 0] - E_gs) * c_cov + H_A1[0, 1] * c_ion
sol = sp.solve(eq1, c_ion)[0]                 # c_ion in terms of c_cov
ratio = sp.simplify(sol / c_cov)
print(f"\nc_ion / c_cov = {ratio}")

# Chirgwin-Coulson weights (orthonormal basis at s=0 -> squared coefficients)
w_cov = sp.simplify(1 / (1 + ratio**2))
w_ion = sp.simplify(ratio**2 / (1 + ratio**2))
print(f"\nw_cov(U, h) = {w_cov}")
print(f"w_ion(U, h) = {w_ion}")

# Simplify with h^2 = t^2
t = sp.Symbol('t', positive=True)
w_cov_t = sp.simplify(w_cov.subs(h**2, t**2))
w_ion_t = sp.simplify(w_ion.subs(h**2, t**2))
print(f"\nWith t^2 = h^2:")
print(f"  w_cov = {w_cov_t}")
print(f"  w_ion = {w_ion_t}")


# ------------------------------------------------------------------------
# 4. The physical story: U-dependent bond character
# ------------------------------------------------------------------------
print("\n" + "=" * 66)
print("H2 ground state vs U (at t = 1, orthogonal orbitals)")
print("=" * 66)
print(f"{'U':>6}  {'E_gs':>10}  {'w_cov':>8}  {'w_ion':>8}  interpretation")
for Uval in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 100.0]:
    E_num = float(E_gs.subs({U_sym: Uval, h: -1}))
    wc = float(w_cov.subs({U_sym: Uval, h: -1}))
    wi = 1 - wc
    tag = ("HF-like (50/50)"       if abs(wc - 0.5) < 0.05 else
           "predominantly covalent" if wc > 0.8 else
           "mixed covalent/ionic")
    print(f"  {Uval:4.1f}  {E_num:+10.6f}  {wc:7.4f}  {wi:7.4f}  {tag}")

print("\nLimits:")
print("  U = 0       w_cov = 1/2, w_ion = 1/2     (restricted HF result)")
print("  U -> infty  w_cov -> 1, w_ion -> 0       (pure Heitler-London)")
print("  crossover at U ~ 4 t  (where the ionic-state penalty equals the resonance)")


# ------------------------------------------------------------------------
# 5. Taylor series for H2 -- for comparison with the benzene case
# ------------------------------------------------------------------------
print("\n" + "=" * 66)
print("Taylor series of E_gs(U) at t = 1 -- compare to benzene's")
print("=" * 66)
series_H2 = sp.series(E_gs.subs(h, -1), U_sym, 0, 9).removeO()
print(f"  H2:        E(U) = {series_H2}")
print(f"  Benzene:   E(U) = -8 + (3/2)U - (29/288) U^2 "
      "- (2855/5971968) U^4 + (855791/61917364224) U^6 + ...")
print("\nH2 has a clean closed form  E(U) = U/2 - sqrt((U/2)^2 + 4 t^2)")
print("because the A_1 block is only 2 x 2.")
print("Benzene has a 38-dim A_1g block -> rational Taylor coefficients but no")
print("elementary closed form; the VB resonance structure is much richer.")
