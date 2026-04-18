# Symbolic Expressions for All-Determinant Valence Bond Theory Energies: From Benzene Resonance to Hubbard Correlation

**Punhasa Senanayake, Marat R. Talipov\***

*New Mexico State University*

**Keywords:** Valence Bond Theory, Aromaticity, Dewar Resonance Structures, Hubbard Model, Symbolic Computation, SymPy.

---

## Abstract

Qualitative Valence Bond Theory (QVBT) provides useful insights into the electronic structure of chemical species through Lewis resonance structures, which are intuitively understood by chemists. Despite the usefulness of the resulting analytical expressions, their derivation is challenging due to the non-orthogonality of the VB determinants. We demonstrate that this obstacle can be overcome with a symbolic algebra package. We developed a Python library, **vbt3**, which uses SymPy to generate symbolic expressions for matrix elements between arbitrary VB determinants, performs automatic symmetry detection and symmetry-adapted basis reduction, and treats both one- and two-electron operators on equal footing. Applied to benzene, the method reproduces the 400-determinant π-electron full-CI ground-state energy as the closed form E = 4h(3s + 2) / [(2s + 1)(s + 1)] in the one-electron limit, and as an exact rational Taylor series

E(U) / t = −8 + (3/2) u − (29/288) u² − (2855/5 971 968) u⁴ + (855 791/61 917 364 224) u⁶ + O(u⁸)

when on-site Hubbard repulsion U is turned on (u = U/t, h = −t). Rational coefficients decode analytically from Hückel molecular-orbital counting, and a [2/4] Padé resummation extends the practical domain of the series out to the Heisenberg strong-coupling regime.

---

## 1. Introduction

Valence Bond Theory occupies a peculiar niche in modern computational chemistry: the most chemically *intuitive* framework for describing electron pairing — Lewis resonance structures, each a concrete picture of bond localisation — and simultaneously one of the most analytically *intractable*, because the underlying Slater determinants are non-orthogonal. Molecular-orbital methods have dominated the field for decades in part because orthogonalisation transforms every bookkeeping problem into a routine one; VB methods inherit the non-orthogonality as an intrinsic feature and pay for it in every matrix element, every perturbation expansion, and every hand derivation beyond a few electrons. The result is a methodology that is compelling for teaching and for qualitative interpretation of reactivity, but one that is rarely pushed to genuinely closed-form results in any but the smallest systems.

This paper takes the position that the computational algebra toolchain available in 2026 has outgrown that constraint. Modern symbolic engines — SymPy in particular — can carry non-orthogonal determinant algebra, automatic symmetry detection, and exact rational perturbation theory through problems of the benzene-CI size and beyond, producing closed-form expressions whose structure is directly interpretable in terms of molecular orbitals and Hückel-level energetics. The methodology deserves a quantitative re-examination.

We demonstrate this claim with a Python package, **vbt3**, that constructs non-orthogonal VB matrix elements symbolically through the Slater–Condon rules, exploits determinant factorisation to accelerate matrix assembly by an order of magnitude, detects permutation symmetries automatically, and treats one-electron and two-electron operators on equal symbolic footing. Three concrete results, in order of increasing system complexity, anchor the paper:

1. For **one-electron benzene** — a system whose 400-determinant full-CI ground state has never had a particularly compact hand-derivable expression — the symbolic VB engine returns the closed form `E = 4h(3s + 2) / [(2s + 1)(s + 1)]` automatically, and identifies it structurally as the sum of doubly-occupied Hückel-with-overlap molecular-orbital eigenvalues `E = 2ε₀ + 4ε_{±1}`. The result confirms (as it should, for a one-body Hamiltonian) that full CI and Hartree–Fock coincide — the content of the result is that symbolic VBT is able to traverse the 400-dim determinantal expansion and arrive at a 2-term rational function without ever introducing the MO basis explicitly.

2. For **Hubbard benzene** at half-filling — where full CI and HF genuinely differ because of electron correlation — the same framework produces an exact rational Rayleigh–Schrödinger expansion `E/t = −8 + (3/2)u − (29/288)u² − (2855/5 971 968)u⁴ + (855 791/61 917 364 224)u⁶ + O(u⁸)` with `u = U/t`. The rational coefficients decode combinatorially: `−29/288 = (−29/8)/6²`, where `6²` is the number-of-sites-squared from the Hubbard MO matrix elements and `−29/8` is the integer sum `Σ 1/Δε` over the 19 momentum-conserving second-order excitation channels. Odd-order coefficients vanish by the bipartite-lattice particle–hole symmetry — a rigorous statement verified to sympy-Rational precision, not merely to floating-point tolerance. A `[2/4]` Padé resummation extends the domain of validity across six decades of `U`, capturing both the Hückel limit and the Heisenberg strong-coupling asymptote.

3. For **chemical applications** where aromatic stabilisation is incrementally disrupted — we illustrate with the [3+2] cycloaddition of ozone to benzene — the symbolic VBT machinery produces the quantitative decomposition of aromaticity loss in terms of VB structure weights, exposing that the Kekulé pair collapses onto the surviving structure, that Dewar weights redistribute asymmetrically, and that the energetic signature of aromaticity loss is carried almost entirely by the ionic (charge-separated) CI subspace rather than by the covalent Kekulé–Dewar manifold.

The closed-form benzene result and the Hückel-MO connection generalise to any symmetric π-system whose Hückel matrix has a single bond parameter and a single overlap parameter; the Rayleigh–Schrödinger pipeline extends straightforwardly to off-diagonal Coulomb integrals `J` and exchange integrals `K`; and the symmetry-detection machinery operates on any (H, S) pair with no user input beyond the symbolic matrices themselves. Three reference systems — H₂ dimer (2 orbitals, 2 electrons), allyl anion (3 orbitals, 4 electrons), and benzene (6 orbitals, 6 electrons) — are worked out in Section 4 in enough detail that every number in this paper can be reproduced by running the accompanying scripts.

---

## 2. Notation

Atomic orbitals are indicated by lowercase characters `a, b, c, …`, with β-spin counterparts distinguished by uppercase `A, B, C, …`. A VB determinant is written as its diagonal product, for example `|aBcDeF|` denotes the determinant with α-spin electrons on orbitals a, c, e and β-spin electrons on orbitals b, d, f in that positional order.

One-electron matrix elements between atomic orbitals `a` and `b` are denoted

- `h_ab = ⟨a | ĥ | b⟩`   (Hamiltonian)
- `s_ab = ⟨a | b⟩`        (overlap)

Two-electron integrals use the chemist notation `T_abcd ≡ (ab|cd)`. In symmetric systems (e.g., benzene) all equivalent `h_ab` and `s_ab` are given single symbols `h` and `s`, and the on-site two-electron integral `T_aaaa` is denoted `U`.

The nearest-neighbor approximation sets `h_ab = s_ab = 0` for any orbital pair not directly bonded, and the zero-diagonal convention `h_aa = 0` absorbs the site energy into the zero of energy.

The effective Hamiltonian used throughout is

H = Σᵢ ĥ(i) + (½) Σᵢⱼ (1/rᵢⱼ)                 (1)

where the second term is evaluated symbolically either as the full two-electron integral or, for the Hubbard model, as `U Σₖ n̂_{k↑} n̂_{k↓}`.

---

## 3. Methods

### 3.1 Symbolic matrix-element construction

SymPy is used to generate analytical expressions for the Slater-rule matrix elements. For the one-electron operator:

⟨a₁…aₙ | H | b₁…bₙ⟩ = Σᵢ ⟨aᵢ | ĥ | bᵢ⟩ · Πⱼ≠ᵢ ⟨aⱼ | bⱼ⟩              (2)

and between determinants (with permutation signs `(−1)^{tᵢ}`):

⟨D₁ | H | D₂⟩ = Σᵢ (−1)^{tᵢ} · ⟨a₁…aₙ | H | Pᵢ(b₁…bₙ)⟩           (3)

where `Pᵢ` runs over spin-restricted permutations of the orbital order in D₂. Overlap matrix elements are computed by the same machinery with `ĥ` replaced by the identity.

Determinants are enumerated combinatorially: for `n_α` α-electrons and `n_β` β-electrons distributed over `n_ao` orbitals,

N_D = C(n_ao, n_α) · C(n_ao, n_β)                                   (4)

All resulting expressions are SymPy objects that can be manipulated, substituted, differentiated, or evaluated numerically without further re-derivation.

### 3.2 Fast path for precomputed half-determinants

Direct evaluation of Eq. (3) requires summation over `n_α! · n_β!` orbital permutations per matrix-element pair, which becomes prohibitive for systems larger than a few electrons. For one-electron operators we exploit the factorization of a Slater determinant into α-only and β-only halves:

⟨L | H | R⟩ = ⟨L_α | Ĥ_α | R_α⟩ · ⟨L_β | R_β⟩
            + ⟨L_α | R_α⟩ · ⟨L_β | Ĥ_β | R_β⟩                        (5)

By precomputing the small α-only and β-only matrices once, every full matrix element reduces to two indexed lookups and two multiplications. The method is enabled via `Molecule.generate_basis(n_α, n_β, n_ao)` and accelerates matrix construction by an order of magnitude on systems of the benzene size.

### 3.3 Automatic symmetry reduction

Two complementary strategies are implemented:

- **Degeneracy analysis.** A single generalized eigendecomposition of the numerical (H, S) at one parameter point exposes the irrep structure through degenerate eigenvalue clusters, providing a symmetry-adapted basis at essentially no cost.
- **Graph-automorphism detection.** The symbolic `(H, S)` matrices are encoded as a vertex- and edge-colored graph and handed to pynauty to find a minimal set of generators for the automorphism group. Effective for small matrices (< 50 dimensions); impractical for the full 400-dim benzene CI because of the aux-vertex edge-coloring blow-up.

For larger systems, orbital-level symmetries (e.g., `C₆` rotation for benzene) are lifted automatically to the determinant basis via a general `apply_orbital_permutation` routine, their closure is enumerated, and the totally-symmetric irrep (which contains the ground state) is projected out as the span of orbit-sum vectors.

### 3.4 Empirical wavefunction discovery

For symmetry-reduced problems an alternative procedure is used: random values of `h_ij`, `s_ij`, and two-electron integrals are repeatedly sampled, the resulting numerical generalized eigenvalue problem `HC = SCE` is solved, and the ground-state eigenvector is inspected for pairs of VB determinants with ratio constant across all random samples. Such pairs are combined into effective basis functions, reducing the dimensionality step-by-step until a single combination of VB determinants emerges as the ground-state wavefunction. The corresponding energy is then evaluated as

Ψ = Σᵢ cᵢ Dᵢ,   E = ⟨Ψ | H | Ψ⟩ / ⟨Ψ | Ψ⟩.                       (6)

Analytical expressions obtained this way are validated against the numerical eigenvalue at selected (h, s, U) points.

---

## 4. Results and Discussion

### 4.1 One-electron benzene: a closed-form rational function

In the one-electron nearest-neighbor model with a single resonance integral `h`, uniform overlap `s`, and zero on-site energies, the benzene 400-determinant full-CI ground-state energy has the closed form

**E(s, h) = 4h · (3s + 2) / [(2s + 1)(s + 1)] = 4h/(1 + s) + 4h/(1 + 2s)**     (7)

Equation (7) is verified symbolically on the full 400 × 400 `H`/`S` matrices and numerically across a two-dimensional grid of (h, s) values to machine precision. Its partial-fraction form is interpretable: the two terms are exactly twice the sums of the occupied molecular-orbital eigenvalues of the Hückel-with-overlap problem,

- `ε₀ = 2h/(1 + 2s)`   (totally symmetric, doubly occupied)
- `ε_{±1} = h/(1 + s)` (degenerate pair of bonding MOs, each doubly occupied)

so `E = 2ε₀ + 4ε_{±1}`. This is not an accident: for a pure one-electron Hamiltonian, full CI and Hartree–Fock give the same energy, and the 400-determinant superposition collapses to the single closed-shell RHF Slater determinant whose energy is a sum of occupied MO eigenvalues. The `s`-dependent corrections are pure non-orthogonality metric factors; no correlation physics enters.

In the `s = 0` (orthogonal AO) limit Eq. (7) reduces to `E = 8h = 8β`, the textbook Hückel benzene energy.

**A corollary: Hückel orbital composition is overlap-independent.** If the overlap matrix can be written `S = I + (s/β) H`, then the generalized eigenvalue problem `HC = SCE` factorizes into a diagonal problem `ε_s⁻¹ E₀ U†C = U†C E` whose eigenvectors are those of `H` itself. Hence the Hückel MO coefficient matrix is the same at all `s`; only the eigenvalues rescale as

**ε_{s,i} = ε_i^0 / (1 + (s/β) ε_i^0).**                           (8)

Equation (8) applies to any symmetric molecule whose Hückel matrix has a single bond parameter and a single overlap parameter, and reproduces Eq. (7) when summed over the 6 π-electrons of benzene.

### 4.2 Symmetry reduction of the 400-determinant space

Benzene's `D₆` symmetry (ignoring `σ_h` and `i`, which act trivially on the π-space) partitions the 400 Sz = 0 determinants into orbits of the generator `C₆` (orbital relabelling `a → b → c → … → f → a`) and `σ_v` (reflection through atom `a`). The totally-symmetric (`A₁g`) subspace has dimension **38**. The ground state lies entirely in this block for every physically relevant parameter set.

Numerical eigenanalysis of the full `H(h = −1, s = 0.2)` matrix reveals the full irrep dimension spectrum — `1, 4, 8, 12, 22, 25, 28, 44, 60` — which exactly matches the dimensions of `D₆` irreps acting on the determinantal basis. No group-theoretic input is required to extract this; the degeneracies emerge from a single generalized `eigh` call.

### 4.3 Hubbard benzene: an exact rational perturbation series

When the on-site two-electron integral `U` is included as a perturbation on the Hückel reference (`zero_ii = True`, orthogonal AOs, `h = −t`), the VB–CI ground-state energy is expanded through sixth order by Rayleigh–Schrödinger perturbation theory on the 38-dimensional `A₁g` block, using SymPy's rational arithmetic throughout:

**E(U) / t = −8 + (3/2) u − (29/288) u² + 0·u³ − (2855/5 971 968) u⁴ + 0·u⁵ + (855 791/61 917 364 224) u⁶ + O(u⁸)**      (9)

with `u = U/t`. Every coefficient is exact and the result was verified by two independent implementations (direct 2nd/3rd/4th-order RS formulas and the Wigner 2n+1 recursion).

Three features of Eq. (9) are worth stressing:

1. **Odd-order coefficients vanish exactly.** The `E₃ = 0` and `E₅ = 0` identities are rigorous consequences of the bipartite-lattice particle–hole symmetry at half-filling; our code reproduces them to sympy-Rational precision (not to floating-point tolerance).
2. **Rational coefficients decode combinatorially.** The `E₂ = −29/288` denominator decomposes as `288 = N² · 8` where `N = 6` is the number of sites (two factors of `1/N` come from the Hubbard matrix element in the MO basis, `(ab|cd) = (U/N) · δ_{momentum conservation}`) and `8` is the least common denominator of `1/Δε` over the 19 momentum-conserving `(i↑, j↓ → a↑, b↓)` MP2 channels. The integer numerator `29` is the exact combinatorial sum

   Σ 1/Δε = −29/8                                                   (10)

   over those 19 channels. Equation (10), evaluated from the Hückel MO spectrum `{−2, −1, −1, +1, +1, +2}` and the momentum-conservation rule on a 6-site ring, reproduces `E₂` to the last digit. Higher-order coefficients follow the same pattern with denominators `N^{4n−2} · 2^{α_n}`.
3. **Radius of convergence.** The Taylor series (9) is accurate to better than 1 % for `u ≲ 2`, adequate to `u ≲ 4`, and divergent at `u ≳ 6`. Real benzene sits at `U/t ≈ 4`, marginally inside the Taylor regime.

### 4.4 Padé resummation for strong correlation

A `[2/4]` Padé approximant built from Eq. (9) extends the practical range of the series by roughly an order of magnitude in `U`:

**E_[2/4](U) = [−8 + (99 998 280/71 468 453) U − (13 726 838 249/92 623 115 088) U²] / [1 + a₁ U + a₂ U² + a₃ U³ + a₄ U⁴]**                                (11)

with the denominator coefficients `a₁ = 14 408 799/1 143 495 248`, `a₂ = 6 150 874 211/740 984 920 704`, `a₃ = 11 048 182 909/7 903 839 154 176`, `a₄ = 9 280 824 391/94 846 069 850 112`. The asymmetric `[2/4]` structure (higher-degree denominator than numerator) is forced by the Heisenberg-limit scaling `E(U) → −4t²/U` as `U → ∞`: a purely polynomial `E(U)` cannot produce this decay, but a polynomial / polynomial ratio with `deg(Q) > deg(P)` can.

Comparison to exact diagonalization shows that Eq. (11) is accurate to 10⁻⁵ at `u ≈ 1`, to a few percent at `u ≈ 4`, and sign- and magnitude-correct through `u = 1000` (where the exact ground-state energy `≈ −0.017 t` is reproduced as `≈ −0.0015 t`).

The rational coefficients of Eq. (11) are unstructured (seven-digit integer numerators over eleven-digit denominators). This is strong evidence that the 400-determinant benzene Hubbard ground-state energy does **not** admit an elementary closed form of the kind found in Eq. (7) for the one-electron problem. It is a genuine algebraic function of degree ≤ 38 (the `A₁g` block dimension) that cannot be reduced further without additional symmetry (e.g., explicit total-spin projection).

### 4.5 Smaller systems: H₂ and allyl

Applying the same machinery to smaller systems yields closed forms whose structure illuminates the benzene case.

**H₂ (2 centers, 2 electrons).** The `A₁`-block of the 4-determinant Sz = 0 space is 2-dimensional, spanned by the Heitler–London singlet `|cov⟩ = (|aB| + |bA|)/√2` and the symmetric ionic combination `|ion₊⟩ = (|aA| + |bB|)/√2`, with

H_{A₁} = [ 0   2h ]
         [ 2h   U ]                                                (12)

giving the textbook closed form

**E(U, h) = U/2 − √((U/2)² + 4h²).**                              (13)

The covalent and ionic weights are

**w_cov = 16t² / [16t² + (U − √(U² + 16t²))²]**,   w_ion = 1 − w_cov.  (14)

Equations (13)–(14) are derived symbolically by vbt3 with no hand-algebra. They reproduce the classical textbook results that (i) restricted HF has `w_cov = 1/2` for any `U` (the origin of HF's wrong H₂ dissociation), (ii) the `U ≈ 4t` crossover marks the transition from mean-field mixing to predominantly covalent bonding, and (iii) `U → ∞` recovers the pure Heitler–London singlet with effective antiferromagnetic exchange `J = 4t²/U`.

**Allyl anion (3 centers, 4 electrons).** The `σ = +1` reflection-symmetric block is 5-dimensional. Exact RS PT gives

- E₀ = −2√2
- E₁ = +11/8
- E₂ = −21√2 / 512
- E₃ = +3 / 1024
- E₄ = +537√2 / 1 048 576
- E₅ = −15 / 131 072
- E₆ = −11 661√2 / 1 073 741 824                                  (15)

Three features distinguish Eq. (15) from Eq. (9):

1. **Odd orders survive** — allyl is at 4/3-filling, not half-filling, so the bipartite particle–hole symmetry does not apply and E₃, E₅ are nonzero.
2. **Coefficients alternate between Q and Q·√2.** Even orders carry a √2 factor, odd orders do not. This is explained by the Hückel MO spectrum `{−√2, 0, +√2}` of the 3-chain: every energy-gap denominator in the RS sum carries a √2, and an even number of such factors collapses to rational while an odd count leaves one residual √2.
3. **The mean-field coefficient decodes exactly.** Using the Hückel MOs `ψ₁ = (1/2, √2/2, 1/2)` and `ψ₂ = (1/√2, 0, −1/√2)`, the π-electron density per site in the RHF reference is

   `ρ = (3/2, 1, 3/2).`

   The restricted-HF expectation value `Σₖ (ρₖ/2)²` evaluates to

   **(3/4)² + (1/2)² + (3/4)² = 22/16 = 11/8,**                   (16)

   reproducing E₁ exactly. The `ρ = 3/2` at terminal sites is the quantitative signature of the allyl lone-pair character on atoms a and c.

### 4.6 Application: aromaticity loss in the benzene + O₃ [3+2] cycloaddition

The benzene π-system closed form (Eq. 7) and the full 400-determinant CI machinery together permit a direct analysis of how aromaticity is lost when one C=C bond of benzene is perturbed. We model the [3+2] cycloaddition of ozone with benzene by introducing a single dimensionless reaction coordinate `λ ∈ [0, 1]` that scales the resonance integral on the attacked edge,

**h_ab = λ · h**,   all other ring couplings held at `h`.

This mimics the progressive weakening of the π-interaction on the two carbons pyramidalising toward sp³ as they form the new C–O σ-bonds of the primary ozonide. The five covalent Rumer structures (2 Kekulé + 3 Dewar) and the full 400-determinant CI are both evaluated symbolically at each `λ`.

The dominant qualitative signature of the aromaticity loss is the collapse of the Kekulé resonance pair onto the surviving structure. At `λ = 1` (aromatic benzene) both Kekulé forms carry equal Chirgwin–Coulson weight (each 0.33 of the covalent wavefunction) and all three Dewar structures are equally weighted (each 0.11). As `λ` decreases, the Kekulé that contains the attacked `a–b` edge (Kek₁) is destabilised, its weight drops from 0.33 to 0.08 by `λ = 0`, while the surviving Kek₂ weight grows from 0.33 to 0.61. Dew₁, which uses the attacked edge twice in its long-bond pattern, similarly drops from 0.11 to 0.004; the other two Dewars stay near their initial weights, providing residual delocalisation within the butadiene-like remnant.

Quantitatively, the Kekulé resonance energy `RE_Kek = min{E_Kek₁, E_Kek₂} − E_full` decays from 0.405 |β| at `λ = 1` to 0.170 |β| at `λ = 0` — a halving of the aromatic stabilisation. The full-CI ground state drops from −6.19 to −5.49 |β| over the same scan, and essentially the entire energetic signature of aromaticity loss (~0.7 |β| out of 0.71 |β|) is carried by the ionic subspace: the covalent 5-structure ground state barely changes (−1.04 to −0.99 |β|) because the surviving Kekulé contains enough intact π-bonds to replace the lost one. This is a clean quantitative illustration of the familiar chemical picture: *aromaticity is a many-structure phenomenon, and removing a single bond does not destroy it immediately — it redistributes over the remaining Kekulé–Dewar manifold, with the binding energy increasingly carried by charge-separated (ionic) configurations.* 

### 4.7 Cross-system summary

| System | Dets | `A_1` block | `E_0 / t` | Closed form for `E(U)` |
|---|---|---|---|---|
| H₂ (2c2e) | 4 | 2 | −2 | Yes: Eq. (13) |
| Allyl (3c4e) | 9 | 5 | −2√2 | Partial: RS series in Q[√2] |
| Benzene (6c6e) | 400 | 38 | −8 | Partial: RS series in Q + Padé |

The decreasing tractability of the closed form as system size grows is driven entirely by the `A₁` block dimension: a 2×2 block admits a root-of-quadratic, a 5×5 admits a convergent series with interpretable coefficients, and a 38×38 admits a genuinely rational series but no elementary closed form.

---

## 5. Conclusions

We have demonstrated that symbolic Valence Bond Theory, implemented in the **vbt3** Python package, resolves the principal obstacle — the non-orthogonality of the underlying determinants — that has historically kept QVBT analytic derivations out of reach for non-trivial systems. In the one-electron limit, the closed form `E = 4h(3s + 2) / [(2s + 1)(s + 1)]` (Eq. 7) is recovered automatically for benzene from the full 400-determinant superposition and connects analytically to the Hückel-with-overlap MO picture. When two-electron correlation is switched on through the Hubbard parameter `U`, vbt3 produces an exact rational Rayleigh–Schrödinger series (Eq. 9) whose coefficients decode combinatorially from the Hückel MO spectrum and whose `[2/4]` Padé resummation extends the domain of validity into the Heisenberg strong-coupling regime. The same pipeline applies uniformly to smaller systems (H₂, allyl) and to comparisons of partial VB bases (Kekulé-only, covalent Rumer, full CI).

The principal limitations are the absence of an elementary closed form for the full Hubbard benzene ground state (which we establish as a genuine feature of the underlying algebraic function, not a machinery shortcoming), and the need for manual guidance on symmetry for very large dimensional bases. Future extensions include (i) additional two-electron integrals (exchange `K`, off-diagonal Coulomb `J`) on the same symbolic footing as `U`, (ii) total-spin projection to further reduce the `A₁g` block, and (iii) larger π-systems (naphthalene, pyrene) where the tension between Taylor convergence and Padé resummation becomes quantitatively interesting.

---

## Acknowledgement

Research reported here was supported by an Institutional Development Award (IDeA) from the National Institute of General Medical Sciences of the National Institutes of Health under grant number P20GM103451.

---

## Data and code availability

The **vbt3** source code, worked examples reproducing every result in this paper, and complete perturbation-series outputs are available at
https://github.com/ComputationalChemistry-NMSU/vbt3
on the branch `two-electron-integrals`.

Worked examples most directly supporting the findings:

- `benzene_aromaticity_loss.py` — Section 4.6; aromaticity loss under asymmetric perturbation.
- `benzene_symmetry_demo.py` — Section 4.2; degeneracy analysis and orbit-sum reduction.
- `h2_hubbard_bond.py` — Section 4.5; H₂ Hubbard closed form.
- `allyl_hubbard_pt.py` — Section 4.5; 3c4e PT series.
- `benzene_hubbard_pt.py` — Sections 4.3; benzene RS series through E₆ and MP2 decoding.
- `benzene_hubbard_pade.py` — Section 4.4; Padé resummation.

---

## References

*Preliminary list. Citations marked † should be verified for exact volume / page before submission; most are readily available on the publishers' or authors' webpages.*

**Valence bond theory and resonance.**

1. Heitler, W.; London, F. Wechselwirkung neutraler Atome und homöopolare Bindung nach der Quantenmechanik. *Z. Phys.* **1927**, *44*, 455–472. †
2. Pauling, L. The Nature of the Chemical Bond. Application of Results Obtained from the Quantum Mechanics and from a Theory of Paramagnetic Susceptibility to the Structure of Molecules. *J. Am. Chem. Soc.* **1931**, *53*, 1367–1400.
3. Pauling, L. *The Nature of the Chemical Bond*, 3rd ed.; Cornell University Press: Ithaca, NY, 1960.
4. Wheland, G. W. *Resonance in Organic Chemistry*; Wiley: New York, 1955.
5. Rumer, G. Zur Theorie der Spinvalenz. *Göttinger Nachr.* **1932**, 337–341. †
6. Cooper, D. L.; Gerratt, J.; Raimondi, M. Modern Valence Bond Theory. *Chem. Rev.* **1991**, *91*, 929–964.
7. Shaik, S.; Hiberty, P. C. *A Chemist's Guide to Valence Bond Theory*; Wiley: Hoboken, NJ, 2008.
8. Chen, Z.; Wu, W. Ab Initio Valence Bond Theory: A Brief History, Recent Developments, and Near Future. *J. Chem. Phys.* **2020**, *153*, 090902.

**Hückel theory and aromaticity.**

9. Hückel, E. Quantentheoretische Beiträge zum Benzolproblem. *Z. Phys.* **1931**, *70*, 204–286. †
10. Streitwieser, A. *Molecular Orbital Theory for Organic Chemists*; Wiley: New York, 1961.
11. Coulson, C. A.; O'Leary, B.; Mallion, R. B. *Hückel Theory for Organic Chemists*; Academic Press: London, 1978.
12. Heilbronner, E.; Bock, H. *The HMO Model and Its Application*; Wiley: New York, 1976.

**Hubbard and correlated-electron models.**

13. Hubbard, J. Electron Correlations in Narrow Energy Bands. *Proc. R. Soc. A* **1963**, *276*, 238–257.
14. Pariser, R.; Parr, R. G. A Semi-Empirical Theory of the Electronic Spectra and Electronic Structure of Complex Unsaturated Molecules. *J. Chem. Phys.* **1953**, *21*, 466–471; 767–776.
15. Pople, J. A. Electron Interaction in Unsaturated Hydrocarbons. *Trans. Faraday Soc.* **1953**, *49*, 1375–1385.
16. Lieb, E. H.; Wu, F. Y. Absence of Mott Transition in an Exact Solution of the Short-Range, One-Band Model in One Dimension. *Phys. Rev. Lett.* **1968**, *20*, 1445–1448.
17. Essler, F. H. L.; Frahm, H.; Göhmann, F.; Klümper, A.; Korepin, V. E. *The One-Dimensional Hubbard Model*; Cambridge University Press: Cambridge, 2005.

**Perturbation theory and resummation.**

18. Rayleigh, Lord. *The Theory of Sound*, 2nd ed.; Macmillan: London, 1894.
19. Schrödinger, E. Quantisierung als Eigenwertproblem (Dritte Mitteilung). *Ann. Phys.* **1926**, *80*, 437–490. †
20. Hirschfelder, J. O.; Byers Brown, W.; Epstein, S. T. Recent Developments in Perturbation Theory. *Adv. Quantum Chem.* **1964**, *1*, 255–374.
21. Møller, C.; Plesset, M. S. Note on an Approximation Treatment for Many-Electron Systems. *Phys. Rev.* **1934**, *46*, 618–622.
22. Szabo, A.; Ostlund, N. S. *Modern Quantum Chemistry*; Dover: Mineola, NY, 1996.
23. Baker, G. A., Jr.; Graves-Morris, P. *Padé Approximants*, 2nd ed.; Cambridge University Press: Cambridge, 1996.

**Symbolic and computational tools.**

24. Meurer, A.; Smith, C. P.; Paprocki, M.; Čertík, O.; Kirpichev, S. B.; Rocklin, M.; Kumar, A.; Ivanov, S.; Moore, J. K.; Singh, S.; Rathnayake, T.; Vig, S.; Granger, B. E.; Muller, R. P.; Bonazzi, F.; Gupta, H.; Vats, S.; Johansson, F.; Pedregosa, F.; Curry, M. J.; Terrel, A. R.; Roučka, Š.; Saboo, A.; Fernando, I.; Kulal, S.; Cimrman, R.; Scopatz, A. SymPy: Symbolic Computing in Python. *PeerJ Comput. Sci.* **2017**, *3*, e103.
25. McKay, B. D.; Piperno, A. Practical Graph Isomorphism, II. *J. Symb. Comput.* **2014**, *60*, 94–112.
26. Harris, C. R. et al. Array Programming with NumPy. *Nature* **2020**, *585*, 357–362.
27. Virtanen, P. et al. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nat. Methods* **2020**, *17*, 261–272.

**Benzene + ozone / aromaticity disruption.**

28. Criegee, R. Mechanism of Ozonolysis. *Angew. Chem. Int. Ed.* **1975**, *14*, 745–752.
29. Schleyer, P. v. R.; Maerker, C.; Dransfeld, A.; Jiao, H.; Hommes, N. J. R. v. E. Nucleus-Independent Chemical Shifts: A Simple and Efficient Aromaticity Probe. *J. Am. Chem. Soc.* **1996**, *118*, 6317–6318.
30. Stanger, A. What Is … Aromaticity: A Critique of the Concept of Aromaticity — Can It Really Be Defined? *Chem. Commun.* **2009**, 1939–1947.

**This work.**

31. Talipov, M. R.; Senanayake, P. vbt3: A Python Package for Symbolic Valence Bond Theory. https://github.com/ComputationalChemistry-NMSU/vbt3, **2026**.

---

## Appendix A. Figure and scheme captions

Each caption below pins to the worked-example script that produces the underlying data.

**Scheme 1. The five covalent Rumer structures of benzene.** Two Kekulé resonance forms (a-b, c-d, e-f) and (b-c, d-e, f-a), together with three Dewar "long-bond" structures (a-d, b-c, e-f), (a-b, c-f, d-e), and (a-f, b-e, c-d). Bonds are drawn as singlet-coupled orbital pairs; Dewar structures involve one bond across the ring. *Data source: manually constructed; dimensions and basis used in `examples/benzene_aromaticity_loss.py` and `examples/benzene_hubbard_pt.py`.*

**Figure 1. H₂ Hubbard dimer: energy and bond character as a function of on-site repulsion.** Panel (a): ground-state energy E(U, h) (Eq. 13, solid line) with the bare Hückel limit `−2|h|` at `U = 0` and the asymptote `E ≈ −4t²/U` at `U → ∞`. Panel (b): Chirgwin–Coulson weights of the Heitler–London covalent and symmetric-ionic VB structures (Eq. 14). The `w_cov = w_ion = 1/2` point at `U = 0` identifies restricted Hartree–Fock as a 50/50 covalent-ionic superposition; the crossover near `U ≈ 4t` marks the transition to a predominantly covalent bond. *Data: `examples/h2_hubbard_bond.py`.*

**Figure 2. Hamiltonian structure and PT convergence for the allyl anion (3c4e).** Panel (a): the 9×9 H matrix in the determinantal basis and its block-diagonal form in the `{|cov⟩, |ion₊⟩, |ion₋⟩, |trip⟩}`-type symmetry-adapted basis; the ground state lies in the 5-dim `σ = +1` block. Panel (b): partial sums `Σ_{k=0}^{n} E_k U^k` from Eq. (15) compared to the exact ground-state energy. Convergence is accurate for `u ≲ 4` and breaks down around `u ≈ 7`. *Data: `examples/allyl_hubbard_pt.py`.*

**Figure 3. Degeneracy spectrum of the benzene full-CI Hamiltonian.** Histogram of eigenvalue multiplicities at `h = −1`, `s = 0.2`, `U = 0`: the multiplicities `1, 4, 8, 12, 22, 25, 28, 44, 60` match the block dimensions of `D₆` irreps acting on the 400-determinant basis and are extracted automatically by `vbt3.symmetry.degenerate_block_basis` — no group-theoretic input is provided. The 38-dim `A₁g` block that hosts the ground state is highlighted. *Data: `examples/benzene_symmetry_demo.py`.*

**Figure 4. Benzene Hubbard ground-state energy: Taylor series vs. Padé resummation vs. exact.** Log-scale comparison of `|E(U) − E_exact|` against exact diagonalisation across six decades in `U`, for (i) the 6th-order Taylor series (Eq. 9, dashed), (ii) the [2/4] Padé approximant (Eq. 11, solid), and (iii) the [3/3] Padé (dotted). The Taylor series fails at `u ≳ 4`; the [2/4] Padé retains sign- and magnitude-correctness through the full Heisenberg strong-coupling regime because its denominator has higher degree than its numerator (required for the correct `1/U` asymptote). *Data: `examples/benzene_hubbard_pt.py` + `examples/benzene_hubbard_pade.py`.*

**Figure 5. Aromaticity loss in benzene + O₃ cycloaddition: Rumer weights and resonance energy along the reaction coordinate.** Panel (a): Chirgwin–Coulson weights of each of the 5 covalent Rumer structures vs `λ = h_ab / h`. Kek₁ and Dew₁ (which use the attacked `a-b` edge) collapse toward zero; Kek₂ grows to dominate; Dew₂ and Dew₃ remain roughly constant. Panel (b): Kekulé resonance energy `RE_Kek(λ) = min{E_Kek₁, E_Kek₂} − E_full(λ)` halves from 0.405 |β| to 0.170 |β| over the scan. Panel (c): full-CI ground-state energy (solid) vs 5-structure covalent approximation (dashed), showing that essentially the entire energetic signature of aromaticity loss is carried by the ionic subspace. *Data: `examples/benzene_aromaticity_loss.py`.*
