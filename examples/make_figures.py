"""
Generate all figures in the manuscript as PDFs.

Each figure pins to a specific worked example (see Appendix A of
manuscript.md).  Data is computed on the fly for the small systems;
for benzene Hubbard (Figure 4) the expensive symbolic H1/S/H2 matrices
are loaded from /tmp/benzene_hubbard_matrices.pkl if present, else
Figure 4 is skipped with a diagnostic message.

Run:
    python3 examples/make_figures.py [out_dir]

Produces fig1_h2.pdf ... fig5_aromaticity.pdf in out_dir (default
./figures).
"""
import os
import sys
import pickle
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import eigh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from vbt3 import Molecule, SlaterDet, FixedPsi, symmetry
from vbt3.fixed_psi import generate_dets

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else 'figures'
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})


# =======================================================================
# Figure 1 - H2 Hubbard: energy and bond character
# =======================================================================
def figure_1_h2():
    m = Molecule(zero_ii=True, interacting_orbs=['ab'],
                 subst={'h': ('H_ab',), 's': ('S_ab',)},
                 subst_2e={'U': ('1111',)}, max_2e_centers=1)
    P = generate_dets(1, 1, 2)
    H = sp.Matrix(m.build_matrix(P, op='H') + m.o2_matrix(P))
    S = sp.Matrix(m.build_matrix(P, op='S'))
    h, s, U = sp.symbols('h s U')
    H0 = H.subs({h: -1, s: 0})

    # Closed form (via 2x2 A_1 block symbolic work)
    U_vals = np.linspace(0, 20, 201)
    t = 1.0
    E_gs = U_vals / 2 - np.sqrt((U_vals / 2) ** 2 + 4 * t ** 2)
    w_cov = 16 * t ** 2 / (16 * t ** 2 + (U_vals - np.sqrt(U_vals ** 2 + 16 * t ** 2)) ** 2)
    w_ion = 1 - w_cov

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))

    ax[0].plot(U_vals, E_gs, 'k-', lw=1.6)
    ax[0].axhline(-2, ls=':', c='C0', lw=1)
    ax[0].text(12, -1.85, r'$-2t$  (Hückel)', color='C0', fontsize=10)
    U_large = np.linspace(5, 20, 100)
    ax[0].plot(U_large, -4 / U_large, ls='--', c='C3', lw=1)
    ax[0].text(14, -0.6, r'$-4t^2/U$  (Heisenberg)', color='C3', fontsize=10)
    ax[0].set_xlabel(r'$U/t$'); ax[0].set_ylabel(r'$E_{\rm gs} / t$')
    ax[0].set_title('(a)  H$_2$ ground-state energy')
    ax[0].grid(alpha=0.3)

    ax[1].plot(U_vals, w_cov, 'C0-', lw=1.6, label=r'$w_{\rm cov}$')
    ax[1].plot(U_vals, w_ion, 'C3-', lw=1.6, label=r'$w_{\rm ion}$')
    ax[1].axhline(0.5, ls=':', c='k', lw=0.8)
    ax[1].axvline(4, ls=':', c='k', lw=0.8)
    ax[1].text(0.4, 0.45, 'RHF 50/50', fontsize=10)
    ax[1].text(4.3, 0.7, r'crossover  $U\sim4t$', fontsize=10)
    ax[1].set_xlabel(r'$U/t$'); ax[1].set_ylabel('weight')
    ax[1].set_title('(b)  VB character of the bond')
    ax[1].legend(loc='center right'); ax[1].grid(alpha=0.3)
    ax[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig1_h2.pdf'))
    plt.close()
    print('  fig1_h2.pdf')


# =======================================================================
# Figure 2 - Allyl: block structure + PT convergence
# =======================================================================
def figure_2_allyl():
    # Build 9x9 allyl H
    m = Molecule(zero_ii=True, interacting_orbs=['ab', 'bc'],
                 subst={'h': ('H_ab', 'H_bc'), 's': ('S_ab', 'S_bc')},
                 subst_2e={'U': ('1111',)}, max_2e_centers=1)
    P = generate_dets(2, 2, 3)
    H = sp.Matrix(m.build_matrix(P, op='H') + m.o2_matrix(P))
    h, s, Usym = sp.symbols('h s U')
    # Numerical H(U, h=-1, s=0) at various U
    H0_num = np.array(H.subs({h: -1, s: 0, Usym: 0}).tolist(), dtype=float)
    V_num  = np.array((H.subs({h: -1, s: 0, Usym: 1}) -
                       H.subs({h: -1, s: 0, Usym: 0})).tolist(), dtype=float)

    # Exact and partial Taylor sums
    U_vals = np.linspace(0, 8, 121)
    coefs = [-2 * np.sqrt(2), 11 / 8, -21 * np.sqrt(2) / 512, 3 / 1024,
             537 * np.sqrt(2) / 1048576, -15 / 131072,
             -11661 * np.sqrt(2) / 1073741824]

    def E_exact(U):
        return eigh(H0_num + U * V_num)[0][0]

    E_ex = np.array([E_exact(u) for u in U_vals])
    E_t = [sum(coefs[k] * U_vals ** k for k in range(order + 1)) for order in [2, 4, 6]]

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))

    # (a)  Heatmap of the full 9x9 at U=2 showing block structure:
    #      permute rows/cols so sigma=+1 dets appear first
    det_strings = [p.dets[0].det_string for p in P]
    def canon(ds):
        fp = SlaterDet(ds).get_sorted()
        return fp.dets[0].det_string, fp.coefs[0]
    sig = {'a': 'c', 'b': 'b', 'c': 'a'}
    perm, signs = symmetry.apply_orbital_permutation(sig, det_strings, canon)
    order = []
    seen = [False] * 9
    for i in range(9):
        if seen[i]: continue
        j = perm[i]
        if j == i:
            order.append(i); seen[i] = True
        else:
            order.append(i); order.append(j); seen[i] = seen[j] = True
    # Build symmetry-adapted matrix via U-transformation with 5+4 blocks
    # (same construction as examples/allyl_hubbard_pt.py)
    U_plus, U_minus = [], []
    seen = [False] * 9
    for i in range(9):
        if seen[i]: continue
        j = perm[i]; sj = signs[i]
        if j == i:
            seen[i] = True
            v = np.zeros(9); v[i] = 1
            (U_plus if sj == 1 else U_minus).append(v)
        else:
            seen[i] = seen[j] = True
            vp = np.zeros(9); vp[i] = 1; vp[j] = sj; vp /= np.linalg.norm(vp)
            vm = np.zeros(9); vm[i] = 1; vm[j] = -sj; vm /= np.linalg.norm(vm)
            U_plus.append(vp); U_minus.append(vm)
    Umat = np.column_stack(U_plus + U_minus)
    H_U2 = H0_num + 2.0 * V_num
    H_sym = Umat.T @ H_U2 @ Umat
    im = ax[0].imshow(H_sym, cmap='RdBu_r',
                      vmin=-np.abs(H_sym).max(), vmax=np.abs(H_sym).max())
    ax[0].axhline(len(U_plus) - 0.5, color='k', lw=1)
    ax[0].axvline(len(U_plus) - 0.5, color='k', lw=1)
    ax[0].set_title(r'(a)  $H$ in $\{\sigma=+1, \sigma=-1\}$ basis  ($U=2$)')
    ax[0].text(2, -0.8, r'$\sigma=+1$  (5×5)', ha='center', fontsize=10)
    ax[0].text(7, -0.8, r'$\sigma=-1$  (4×4)', ha='center', fontsize=10)
    plt.colorbar(im, ax=ax[0], fraction=0.045, pad=0.03)

    # (b) Taylor convergence
    ax[1].plot(U_vals, E_ex, 'k-', lw=1.8, label='exact')
    for order, (E, ls) in enumerate(zip(E_t, ['--', '-.', ':']), start=2):
        ax[1].plot(U_vals, E, ls=ls, lw=1.2, label=f'Taylor order {2 * order}')
    ax[1].set_xlabel(r'$U/t$'); ax[1].set_ylabel(r'$E / t$')
    ax[1].set_title('(b)  Allyl PT convergence')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    ax[1].set_ylim(-3.5, 2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig2_allyl.pdf'))
    plt.close()
    print('  fig2_allyl.pdf')


# =======================================================================
# Figure 3 - benzene FCI degeneracy spectrum
# =======================================================================
def figure_3_degeneracy():
    m = Molecule(zero_ii=True,
                 subst={'s': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af'),
                        'h': ('H_ab', 'H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af')},
                 interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'])
    m.generate_basis(3, 3, 6)
    H = m.build_matrix(m.basis, op='H')
    S = m.build_matrix(m.basis, op='S')
    h, s = sp.symbols('h s')
    Hn = np.array(H.subs({h: -1, s: 0.2}).tolist(), dtype=float)
    Sn = np.array(S.subs({h: -1, s: 0.2}).tolist(), dtype=float)

    evals, evecs, blocks = symmetry.degenerate_block_basis(Hn, Sn, tol=1e-6)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    degs = [len(b[1]) for b in blocks]
    Es = [b[0] for b in blocks]
    colors = ['C3' if d == 1 else 'C0' for d in degs]
    ax.bar(range(len(blocks)), degs, color=colors, edgecolor='k', lw=0.6)
    for i, (E, d) in enumerate(zip(Es, degs)):
        ax.text(i, d + 1.5, f'{d}', ha='center', fontsize=9)
        ax.text(i, -3.5, f'{E:.2f}', ha='center', fontsize=8, rotation=45)
    ax.set_xlabel('irrep cluster (increasing energy →)')
    ax.set_ylabel('degeneracy')
    ax.set_title(r'Figure 3. Benzene FCI spectrum at $h=-1$, $s=0.2$,  $U=0$'
                 '\n' + 'Highlighted: non-degenerate (1D) clusters '
                 r'include the ground state ($A_{1g}$).')
    ax.set_ylim(-5, max(degs) + 8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig3_degeneracy.pdf'))
    plt.close()
    print('  fig3_degeneracy.pdf')


# =======================================================================
# Figure 4 - benzene Hubbard Taylor vs Pade vs exact
# =======================================================================
def figure_4_pade():
    CACHE = '/tmp/benzene_hubbard_matrices.pkl'
    if not os.path.exists(CACHE):
        print('  fig4: skipping (need /tmp/benzene_hubbard_matrices.pkl; '
              'run examples/benzene_hubbard_pt.py once to generate)')
        return
    with open(CACHE, 'rb') as f:
        H1, S, H2 = pickle.load(f)
    h, s, U = sp.symbols('h s U')
    H = sp.Matrix(H1 + H2).subs({h: -1, s: 0})
    H0 = np.array(H.subs(U, 0).tolist(), dtype=float)
    V = np.array((H.subs(U, 1) - H.subs(U, 0)).tolist(), dtype=float)

    def E_exact(Uv):
        return eigh(H0 + Uv * V)[0][0]

    # Taylor
    C = [sp.Rational(-8), sp.Rational(3, 2), sp.Rational(-29, 288),
         sp.Rational(0), sp.Rational(-2855, 5971968), sp.Rational(0),
         sp.Rational(855791, 61917364224)]
    Cf = [float(c) for c in C]

    # Pade using the same solver as examples/benzene_hubbard_pade.py
    def pade(c, n, m):
        x = sp.Symbol('x')
        A = sp.zeros(m, m); rhs = sp.zeros(m, 1)
        for k in range(n + 1, n + m + 1):
            row = k - (n + 1); rhs[row, 0] = -c[k]
            for j in range(1, m + 1):
                if 0 <= k - j < len(c): A[row, j - 1] = c[k - j]
        b = A.solve(rhs)
        Q = 1 + sum(b[j - 1] * x ** j for j in range(1, m + 1))
        f = sum(c[k] * x ** k for k in range(n + m + 1))
        P_full = sp.expand(f * Q)
        P = sum(P_full.coeff(x, k) * x ** k for k in range(n + 1))
        return sp.lambdify(x, P / Q, 'numpy')

    pade24 = pade(C, 2, 4)
    pade33 = pade(C, 3, 3)

    U_vals = np.geomspace(0.1, 500, 80)
    E_ex = np.array([E_exact(u) for u in U_vals])
    E_t6 = sum(Cf[k] * U_vals ** k for k in range(7))
    E_p24 = np.array([float(pade24(u)) for u in U_vals])
    E_p33 = np.array([float(pade33(u)) for u in U_vals])

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))

    ax[0].plot(U_vals, E_ex, 'k-', lw=1.8, label='exact')
    ax[0].plot(U_vals, E_t6, 'C0--', lw=1.2, label='Taylor(6)')
    ax[0].plot(U_vals, E_p24, 'C3-', lw=1.2, label='Padé [2/4]')
    ax[0].plot(U_vals, E_p33, 'C2:', lw=1.2, label='Padé [3/3]')
    ax[0].set_xscale('log')
    ax[0].set_xlabel(r'$U/t$'); ax[0].set_ylabel(r'$E / t$')
    ax[0].set_title('(a)  Benzene Hubbard ground-state energy')
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3, which='both')
    ax[0].set_ylim(-9, 10)

    err_t6  = np.abs(E_t6 - E_ex)
    err_p24 = np.abs(E_p24 - E_ex)
    err_p33 = np.abs(E_p33 - E_ex)
    for arr in (err_t6, err_p24, err_p33):
        arr[arr == 0] = np.nan
    ax[1].loglog(U_vals, err_t6, 'C0--', lw=1.2, label='Taylor(6)')
    ax[1].loglog(U_vals, err_p24, 'C3-', lw=1.2, label='Padé [2/4]')
    ax[1].loglog(U_vals, err_p33, 'C2:', lw=1.2, label='Padé [3/3]')
    ax[1].set_xlabel(r'$U/t$'); ax[1].set_ylabel(r'$|E_{\rm approx} - E_{\rm exact}|$')
    ax[1].set_title('(b)  absolute error, log-log')
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig4_pade.pdf'))
    plt.close()
    print('  fig4_pade.pdf')


# =======================================================================
# Figure 5 - aromaticity loss under a-b attack
# =======================================================================
def figure_5_aromaticity():
    m = Molecule(
        zero_ii=True,
        subst={'s': ('S_ab', 'S_bc', 'S_cd', 'S_de', 'S_ef', 'S_af'),
               'h': ('H_bc', 'H_cd', 'H_de', 'H_ef', 'H_af')},   # H_ab free
        interacting_orbs=['ab', 'bc', 'cd', 'de', 'ef', 'af'])
    PARENT = 'aBcDeF'
    rumer = [
        FixedPsi(PARENT, coupled_pairs=[(0, 1), (2, 3), (4, 5)]),
        FixedPsi(PARENT, coupled_pairs=[(0, 5), (1, 2), (3, 4)]),
        FixedPsi(PARENT, coupled_pairs=[(0, 1), (2, 5), (3, 4)]),
        FixedPsi(PARENT, coupled_pairs=[(0, 3), (1, 2), (4, 5)]),
        FixedPsi(PARENT, coupled_pairs=[(0, 5), (1, 4), (2, 3)])]
    names = ['Kek$_1$', 'Kek$_2$', 'Dew$_1$', 'Dew$_2$', 'Dew$_3$']

    Hs = m.build_matrix(rumer, op='H')
    Ss = m.build_matrix(rumer, op='S')
    h, s = sp.symbols('h s')
    Hab = sp.Symbol('H_ab')
    H_VAL, S_VAL = -1.0, 0.2
    lambdas = np.linspace(1.0, 0.0, 41)
    weights = np.zeros((5, len(lambdas)))
    Es = np.zeros((5, len(lambdas)))
    E_full = np.zeros(len(lambdas))
    for i, lam in enumerate(lambdas):
        subs = {h: H_VAL, s: S_VAL, Hab: lam * H_VAL}
        Hn = np.array(Hs.subs(subs).tolist(), dtype=float)
        Sn = np.array(Ss.subs(subs).tolist(), dtype=float)
        evals, evecs = eigh(Hn, Sn)
        E_full[i] = evals[0]
        c0 = evecs[:, 0]
        weights[:, i] = c0 * (Sn @ c0)
        Es[:, i] = [Hn[j, j] / Sn[j, j] for j in range(5)]

    RE_Kek = np.min(Es[:2], axis=0) - E_full

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    ls = ['-', '-', '--', '--', '--']
    for j in range(5):
        ax[0].plot(lambdas, weights[j], ls[j], color=colors[j], lw=1.5, label=names[j])
    ax[0].set_xlabel(r'$\lambda = h_{ab}/h$ (reaction coordinate)')
    ax[0].set_ylabel('Chirgwin–Coulson weight')
    ax[0].set_title('(a)  Rumer structure weights')
    ax[0].legend(fontsize=9, ncol=2, loc='upper left')
    ax[0].grid(alpha=0.3); ax[0].invert_xaxis()

    ax[1].plot(lambdas, RE_Kek, 'k-', lw=1.8)
    ax[1].set_xlabel(r'$\lambda$')
    ax[1].set_ylabel(r'$RE_{\rm Kek}$  (units of $|\beta|$)')
    ax[1].set_title(r'(b)  Kekulé resonance energy')
    ax[1].grid(alpha=0.3); ax[1].invert_xaxis()
    ax[1].annotate('aromatic', xy=(1.0, RE_Kek[0]), xytext=(0.8, 0.42),
                   fontsize=10, ha='center')
    ax[1].annotate('broken bond', xy=(0.0, RE_Kek[-1]), xytext=(0.2, 0.22),
                   fontsize=10, ha='center')

    ax[2].plot(lambdas, E_full, 'k-', lw=1.8, label='5-Rumer covalent')
    ax[2].set_xlabel(r'$\lambda$')
    ax[2].set_ylabel(r'$E  /  |\beta|$')
    ax[2].set_title(r'(c)  covalent ground-state energy')
    ax[2].grid(alpha=0.3); ax[2].invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'fig5_aromaticity.pdf'))
    plt.close()
    print('  fig5_aromaticity.pdf')


# =======================================================================
if __name__ == '__main__':
    print(f'writing figures to {OUTDIR}/ ...')
    figure_1_h2()
    figure_2_allyl()
    figure_3_degeneracy()
    figure_4_pade()
    figure_5_aromaticity()
    print('done.')
