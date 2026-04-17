"""
Automatic symmetry detection and symmetry-adapted basis construction.

Two detection strategies:

1. `degenerate_block_basis` — numerical eigenanalysis.
   Diagonalise a single (H, S) numerically and group the eigenvectors by
   degenerate eigenvalue. Each degenerate cluster spans an irrep subspace.
   Fully automatic, no group theory, immediately useful but only exposes the
   block structure at the specific parameter point used.

2. `detect_permutation_group` — graph-automorphism detection via pynauty.
   Encode the symbolic (H, S) as a vertex- and edge-coloured graph; pynauty
   (via Weisfeiler–Leman refinement + backtracking) returns a minimal set of
   generators for the group of basis permutations that preserve the matrices
   symbolically. Requires the optional dependency `pynauty`.

Also provided:

    generate_group(generators, N)       — enumerate a group from generators.
    totally_symmetric_basis(generators, N) — orbit-sum basis for the trivial
        irrep; any H commuting with the generators block-diagonalises here,
        and the ground state of benzene-like problems lives in this block.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.linalg import eigh


def degenerate_block_basis(H_num, S_num=None, tol=1e-8):
    """
    Diagonalise (H, S) and group eigenvectors by degenerate eigenvalue.

    Parameters
    ----------
    H_num : ndarray of shape (N, N).
    S_num : ndarray of shape (N, N), optional. Generalized overlap metric.
    tol   : eigenvalue degeneracy tolerance.

    Returns
    -------
    evals  : (N,) ndarray of sorted eigenvalues.
    evecs  : (N, N) ndarray; columns are eigenvectors, already a
             symmetry-adapted basis up to within-block unitary freedom.
    blocks : list of (eigenvalue, column_indices) pairs, one per irrep cluster.
    """
    if S_num is None:
        evals, evecs = np.linalg.eigh(H_num)
    else:
        evals, evecs = eigh(H_num, S_num)

    blocks = []
    i = 0
    while i < len(evals):
        j = i + 1
        while j < len(evals) and abs(evals[j] - evals[i]) < tol:
            j += 1
        blocks.append((float(evals[i]), list(range(i, j))))
        i = j
    return evals, evecs, blocks


def _canon(expr):
    """Canonical hashable key for a sympy expression.

    Sympy `Expr` objects are already canonically comparable (equal expressions
    hash to the same value), so we use the expression itself wrapped for
    robustness. We only fall back to `sp.expand` when two expressions that
    might be equal don't hash the same.
    """
    return sp.sympify(expr)


def detect_permutation_group(H_sym, S_sym=None):
    """
    Find every basis permutation that leaves H_sym (and S_sym if given)
    unchanged as a symbolic expression, using pynauty.

    The matrices are encoded as a colored graph:
        vertex color of i = (H[i,i], S[i,i])
        edge color of (i,j) = (H[i,j], S[i,j])
    Edge colors are converted to vertex colors via auxiliary vertices.

    Parameters
    ----------
    H_sym : sympy Matrix of shape (N, N).
    S_sym : sympy Matrix of shape (N, N), optional.

    Returns
    -------
    generators : list of ndarray of shape (N,). A generating set for the
                 automorphism group; each array is a permutation of 0..N-1.
    group_order: int. Order of the full group.
    """
    try:
        import pynauty
    except ImportError as e:
        raise ImportError(
            "pynauty is required for detect_permutation_group. "
            "Install with `pip install pynauty` (may need `--no-binary :all:` "
            "on older CPUs without AVX2)."
        ) from e

    if not isinstance(H_sym, sp.Matrix):
        H_sym = sp.Matrix(H_sym)
    if S_sym is not None and not isinstance(S_sym, sp.Matrix):
        S_sym = sp.Matrix(S_sym)

    N = H_sym.shape[0]

    def vcol(i):
        if S_sym is None:
            return (_canon(H_sym[i, i]),)
        return (_canon(H_sym[i, i]), _canon(S_sym[i, i]))

    def ecol(i, j):
        if S_sym is None:
            return (_canon(H_sym[i, j]),)
        return (_canon(H_sym[i, j]), _canon(S_sym[i, j]))

    zero_key = ((_canon(0),) if S_sym is None
                else (_canon(0), _canon(0)))

    # Auxiliary vertex per non-zero edge, grouped by edge-color
    edge_class = {}
    edge_aux = {}
    aux = N
    for i in range(N):
        for j in range(i + 1, N):
            c = ecol(i, j)
            if c == zero_key:
                continue
            edge_aux[(i, j)] = aux
            edge_class.setdefault(c, []).append(aux)
            aux += 1

    total = aux

    g = pynauty.Graph(total)
    for (i, j), a in edge_aux.items():
        g.connect_vertex(a, [i, j])

    vclass = {}
    for i in range(N):
        vclass.setdefault(vcol(i), []).append(i)

    partition = [set(v) for v in vclass.values()] + [set(e) for e in edge_class.values()]
    g.set_vertex_coloring(partition)

    gens_raw, order, *_ = pynauty.autgrp(g)

    # Trim each generator to the first N entries. pynauty should keep aux
    # vertices within their own color class, so original vertices permute
    # among themselves.
    generators = []
    for gen in gens_raw:
        perm = np.asarray(gen[:N], dtype=int)
        if set(perm.tolist()) == set(range(N)):
            generators.append(perm)
    return generators, int(order)


def _compose(p, q):
    """Permutation composition (p o q)(i) = p[q[i]]."""
    return np.asarray([p[q[i]] for i in range(len(p))], dtype=int)


def generate_group(generators, N=None):
    """
    Enumerate every permutation in the group generated by `generators`.
    Returns a list of ndarrays.
    """
    if N is None:
        N = len(generators[0]) if generators else 0
    identity = tuple(range(N))
    seen = {identity}
    queue = [np.arange(N)]
    elements = [np.arange(N)]
    while queue:
        g = queue.pop()
        for gen in generators:
            new = _compose(gen, g)
            key = tuple(new.tolist())
            if key not in seen:
                seen.add(key)
                queue.append(new)
                elements.append(new)
    return elements


def apply_orbital_permutation(orbital_map, basis_dets, canon_fn):
    """
    Induce a basis permutation from an orbital-label permutation.

    Given a permutation of single-particle orbital labels (e.g. C_6 on
    {a,b,c,d,e,f} -> {b,c,d,e,f,a}) and a list of basis determinants,
    compute which basis index each det maps to and the associated fermion
    sign (from re-canonicalising the permuted string). Returns None if the
    permutation does not preserve the basis set (i.e. a permuted det is
    not representable in the given basis).

    Parameters
    ----------
    orbital_map : dict mapping each lowercase label to its image.
                  Uppercase (beta) images are derived automatically.
    basis_dets  : list of strings, one per basis det, in canonical form.
    canon_fn    : callable taking a det_string, returning (canonical_string,
                  sign). Typically wraps SlaterDet(s).get_sorted().

    Returns
    -------
    perm : ndarray of shape (N,). Basis index `i` maps to index `perm[i]`.
    signs: ndarray of shape (N,) with +/- 1 fermion signs.
    """
    lower = list(orbital_map.keys())
    upper_map = {k.upper(): v.upper() for k, v in orbital_map.items()}
    translate = str.maketrans({**orbital_map, **upper_map})

    index_of = {d: i for i, d in enumerate(basis_dets)}
    N = len(basis_dets)
    perm = np.empty(N, dtype=int)
    signs = np.empty(N, dtype=int)
    for i, d in enumerate(basis_dets):
        image = d.translate(translate)
        canon, sgn = canon_fn(image)
        if canon not in index_of:
            return None, None
        perm[i] = index_of[canon]
        signs[i] = sgn
    return perm, signs


def totally_symmetric_basis(generators, N):
    """
    Construct the orbit-sum basis for the trivial irrep: a column for each
    orbit, uniform weight 1/sqrt(orbit_size) on the orbit members, zero
    elsewhere. Columns are orthogonal (different orbits don't overlap) and
    unit-normalised.

    Any H commuting with every generator is block-diagonal in the resulting
    decomposition; this function returns only the totally-symmetric block,
    which contains the ground state of benzene-like problems.

    Parameters
    ----------
    generators : list of ndarray permutations of 0..N-1.
    N          : int.

    Returns
    -------
    U      : ndarray of shape (N, k) where k is the number of orbits.
    orbits : list of index lists.
    """
    if not generators:
        U = np.eye(N)
        return U, [[i] for i in range(N)]

    elements = generate_group(generators, N)
    assigned = [False] * N
    orbits = []
    for i in range(N):
        if assigned[i]:
            continue
        orb = set()
        for g in elements:
            orb.add(int(g[i]))
        for j in orb:
            assigned[j] = True
        orbits.append(sorted(orb))

    U = np.zeros((N, len(orbits)))
    for k, orb in enumerate(orbits):
        for idx in orb:
            U[idx, k] = 1.0
        U[:, k] /= np.sqrt(len(orb))
    return U, orbits
