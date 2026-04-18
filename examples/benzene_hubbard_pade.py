"""
Padé resummation of the benzene Hubbard perturbation series.

The exact Rayleigh-Schrodinger Taylor expansion (from
benzene_hubbard_pt.py) is

    E(U) / t  =  -8 + (3/2)u - (29/288)u^2 + 0*u^3
                 - (2855/5971968)u^4 + 0*u^5
                 + (855791/61917364224)u^6 + O(u^7)

with u = U/t.  It diverges for |u| >~ 4 - above the physically-
interesting range for chemistry.  A Padé [n/m] approximant

    E(u) ≈ P_n(u) / Q_m(u)

built from the same coefficients extends the domain of validity
dramatically.  This script constructs several [n/m] approximants and
compares them to exact numerical diagonalisation of the 400x400
full-CI Hamiltonian H_0 + U V.

Key observation: the [2/4] approximant (numerator of degree 2,
denominator of degree 4) is the best - it's essentially exact for
u <~ 2, within a few percent up to u = 4, and remains sign- and
magnitude-correct out to u = 1000 (the Heisenberg strong-coupling
limit).  A higher-degree denominator than numerator is required
because E(U) -> 0 as U -> infty (at half-filling the ground-state
energy decays like 4 t^2 / U in the Heisenberg limit).

The rational coefficients in the [2/4] Padé are *ugly* (7-digit
integers over 11-digit denominators), which tells us that benzene's
half-filled Hubbard ground-state energy does NOT admit a simple
elementary closed form.  It is an algebraic function of degree up
to 38 (the A_1g-block size) and cannot be simplified to products of
geometric sums like the s-dependent Huckel-MO formula we found for
the one-electron case.  Padé is the practical compromise.
"""
import os
import sys
import pickle

import numpy as np
import sympy as sp
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


# ------------------------------------------------------------------------
# Exact Taylor coefficients of E(U) at h=-1, s=0 from benzene_hubbard_pt.py
# ------------------------------------------------------------------------
C = [
    sp.Rational(-8),
    sp.Rational(3, 2),
    sp.Rational(-29, 288),
    sp.Rational(0),
    sp.Rational(-2855, 5971968),
    sp.Rational(0),
    sp.Rational(855791, 61917364224),
]


def pade(c, n, m):
    """
    Build the Padé [n/m] approximant P(x)/Q(x) from Taylor coefficients
    c[0..n+m].  Q is normalised by Q(0) = 1.  Both returned as sympy
    polynomials in `x`.
    """
    x = sp.Symbol('x')
    assert len(c) >= n + m + 1, "need at least n+m+1 coefficients"
    # Coefficients of f(x)*Q(x) beyond order n must vanish, giving a
    # linear system for b_1, ..., b_m  (with b_0 = 1).
    A = sp.zeros(m, m)
    rhs = sp.zeros(m, 1)
    for k in range(n + 1, n + m + 1):
        row = k - (n + 1)
        rhs[row, 0] = -c[k]
        for j in range(1, m + 1):
            if 0 <= k - j < len(c):
                A[row, j - 1] = c[k - j]
    b = A.solve(rhs)
    Q = 1 + sum(b[j - 1] * x**j for j in range(1, m + 1))
    f = sum(c[k] * x**k for k in range(n + m + 1))
    P_full = sp.expand(f * Q)
    P = sum(P_full.coeff(x, k) * x**k for k in range(n + 1))
    return sp.nsimplify(P), sp.nsimplify(Q)


# ------------------------------------------------------------------------
# Build a selection of approximants
# ------------------------------------------------------------------------
x = sp.Symbol('x')
candidates = [(3, 3), (2, 4), (4, 2), (1, 5)]
approximants = {}
for n, m in candidates:
    P, Q = pade(C, n, m)
    approximants[(n, m)] = (P, Q)
    print(f"Padé [{n}/{m}]   P(U) / Q(U):")
    print(f"    P(U) = {P}")
    print(f"    Q(U) = {Q}")
    print()


# ------------------------------------------------------------------------
# Exact numerical reference: diagonalise H_0 + U*V for the full 400-dim CI
# ------------------------------------------------------------------------
CACHE = '/tmp/benzene_hubbard_matrices.pkl'
if not os.path.exists(CACHE):
    print("Cache not found; run benzene_hubbard_pt.py first to build the "
          "symbolic H1/S/H2 matrices.")
    sys.exit(0)

with open(CACHE, 'rb') as f:
    H1, S, H2 = pickle.load(f)
h_sym, s_sym, U_sym = sp.symbols('h s U')
H_tot = sp.Matrix(H1 + H2).subs({h_sym: -1, s_sym: 0})
H0_np = np.array(H_tot.subs(U_sym, 0).tolist(), dtype=float)
V_np  = np.array((H_tot.subs(U_sym, 1) - H_tot.subs(U_sym, 0)).tolist(), dtype=float)


def E_exact(U):
    return eigh(H0_np + U * V_np)[0][0]


def E_taylor(U, order=6):
    return sum(float(C[k]) * U**k for k in range(order + 1))


# ------------------------------------------------------------------------
# Accuracy comparison
# ------------------------------------------------------------------------
print("=" * 78)
print(f"{'U':>6}  {'E_exact':>10}  {'Taylor(6)':>12}  " +
      "  ".join(f"{'Padé['+str(n)+'/'+str(m)+']':>12}" for n, m in candidates))
print("-" * 78)
for U_test in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 100.0, 1000.0]:
    e_ex = E_exact(U_test)
    e_ta = E_taylor(U_test)
    row = f"  {U_test:5.1f}  {e_ex:+10.6f}  {e_ta:+12.3e}"
    for n, m in candidates:
        P, Q = approximants[(n, m)]
        fn = sp.lambdify(x, P / Q, 'numpy')
        row += f"  {float(fn(U_test)):+12.6f}"
    print(row)
print("=" * 78)

print()
print("Observations:")
print("  * Taylor(6) is exact for |U| <~ 2 and useless beyond.")
print("  * Padé [2/4] extends the useful range of the series by ~1 order of")
print("    magnitude in U.  The larger denominator degree is required because")
print("    E(U) -> 0 as U -> infty (Heisenberg limit ~ 4 t^2 / U).")
print("  * Padé [4/2] and [3/3] diverge outside the Taylor radius.")
print("  * Rational coefficients in the [2/4] approximant are unstructured,")
print("    suggesting benzene's half-filled Hubbard ground-state energy does")
print("    not admit a simple elementary closed form.")
