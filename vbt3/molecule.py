from functools import lru_cache

import numpy
import sympy as sp
from scipy.stats import rankdata

from vbt3.functions import attempt_int, standardize_det, sort_ind, simplify_matrix
from vbt3.numerical import get_coupled
from vbt3.fixed_psi import FixedPsi, generate_dets
from vbt3.numerical import get_combined_from_dict


@lru_cache(maxsize=None)
def _cached_symbol(name):
    return sp.Symbol(name)


_SP_ZERO = sp.Integer(0)
_SP_ONE = sp.Integer(1)


class Molecule:
    # Contains molecule-specific information
    def __init__(self, symm_offdiagonal=True, normalized_basis_orbs=True,
                 interacting_orbs=None, subst=None, zero_ii=True,
                 subst_2e=None, max_2e_centers=4):
        """
        subst contains a list of substitutions to be made, eg ['S':('S_ab','S_bc','S_cd'),'H':('H_ab','H_bc')]
        zero_ii=True sets all H_ii terms to zero
        interacting_orbs is a list of two-letter lowercase strings, eg ['ab','bc','ad'].
        Only these orbital pairs have non-zero integrals
        symm_offdiagonal = True; symmetric matrix
        normalized_basis_orbs = True; S_ii = 1
        """

        self.symm_offdiagonal = symm_offdiagonal
        self.normalized_basis_orbs = normalized_basis_orbs
        self.interacting_orbs = interacting_orbs  # list of two-letter lowercase strings, eg ['ab','bc','ad']

        self.subst = {}
        self.subst_2e = {}

        self.basis = None
        self.basis_a, self.basis_b = None, None
        self.aH, self.aS = None, None
        self.bH, self.bS = None, None
        self.lookup_a, self.lookup_b = {}, {}
        self.precalculated_half_dets = False

        if subst is None:
            subst = {}
        self.parse_subst(subst)

        if subst_2e is None:
            subst_2e = {}
        self.parse_subst_2e(subst_2e)

        self.zero_ii = zero_ii
        self.max_2e_centers = max_2e_centers

        self._o1_expr_cache = {}

    def generate_basis(self, Na, Nb, Norbs):

        self.precalculated_half_dets = False

        self.basis = generate_dets(Na, Nb, Norbs)

        self.basis_a = generate_dets(Na, 0, Norbs)
        for i in range(len(self.basis_a)):
            self.lookup_a[self.basis_a[i].dets[0].det_string] = i

        if Na == Nb:
            self.basis_b, self.lookup_b = self.basis_a, self.lookup_a
        else:
            self.basis_b = generate_dets(Nb, 0, Norbs) # all lookups will be by lower case
            for i in range(len(self.basis_b)):
                self.lookup_b[self.basis_b[i].dets[0].det_string] = i

        self.aH = self.build_matrix(self.basis_a, op='H')
        self.aS = self.build_matrix(self.basis_a, op='S')
        if Na == Nb:
            self.bH, self.bS = self.aH, self.aS
        else:
            self.bH = self.build_matrix(self.basis_b, op='H')
            self.bS = self.build_matrix(self.basis_b, op='S')
        self.precalculated_half_dets = True

    def parse_subst(self, subst):
        for k, v in subst.items():
            if isinstance(v, str):
                self.subst[v] = k
            else:
                for s in v:
                    self.subst[s] = k

    def parse_subst_2e(self, subst_2e):
        for k, v in subst_2e.items():
            if isinstance(v, str):
                self.subst_2e[v] = k
            else:
                for s in v:
                    self.subst_2e[s] = k

    def get_o1_name(self, a, b, o):
        # sort two orbital indices in the alphabetic order
        if self.symm_offdiagonal and (a > b):
            a, b = b, a

        # If only certain orbitals are allowed to interact,
        # check if the orbital pair is in the allowed list
        if self.interacting_orbs is not None and (a != b):
            if not (a + b) in self.interacting_orbs:
                return '0'  # non-interacting orbitals will always give 0 in any term of the direct product

        # Replace terms S_xx by 1 if allowed
        if self.normalized_basis_orbs and (a == b) and (o == 'S'):
            return '1'

        # replace the site energies H_ii by zero if allowed
        if self.zero_ii and (a == b) and (o == 'H'):
            return '0'

        s = '%s_%s%s' % (o, a, b)

        # substitute certain AO matrix elements if needed
        if s in self.subst:
            s = self.subst[s]

        return s

    def get_o1_expr(self, a, b, o):
        # Cached sympy expression for a one-electron AO matrix element
        key = (a, b, o)
        cached = self._o1_expr_cache.get(key)
        if cached is not None:
            return cached
        name = self.get_o1_name(a, b, o)
        if name == '0':
            expr = _SP_ZERO
        elif name == '1':
            expr = _SP_ONE
        else:
            expr = _cached_symbol(name)
        self._o1_expr_cache[key] = expr
        return expr

    def Op_Hartree_product(self, L_orbs, R_orbs, op='H'):
        # Computes a matrix element for two orbital products, e.g <A(1)b(2)...|O|A(1)b(2)...>.
        # Returns a sympy expression.

        nL = len(L_orbs)
        nR = len(R_orbs)

        if nL != nR:
            return _SP_ZERO

        if nL == 0:
            return _SP_ONE if op == 'S' else _SP_ZERO

        lL = L_orbs.lower()
        lR = R_orbs.lower()

        sum_terms = []
        for i_op in range(nL):
            prod_factors = []
            term_zero = False
            for j in range(nL):
                o = op if i_op == j else 'S'
                s = self.get_o1_expr(lL[j], lR[j], o)
                if s is _SP_ZERO:
                    term_zero = True
                    break
                if s is _SP_ONE:
                    continue
                prod_factors.append(s)

            if term_zero:
                elem = _SP_ZERO
            elif not prod_factors:
                elem = _SP_ONE
            elif len(prod_factors) == 1:
                elem = prod_factors[0]
            else:
                elem = sp.Mul(*prod_factors)

            if op == 'S':
                # All Hartree products in <L|S|R> are identical; one is enough.
                return elem

            if elem is not _SP_ZERO:
                sum_terms.append(elem)

        if not sum_terms:
            return _SP_ZERO
        if len(sum_terms) == 1:
            return sum_terms[0]
        return sp.Add(*sum_terms)

    op_orbprod = Op_Hartree_product

    def op_det(self, L, R, op='H'):
        # Returns the matrix element < L | O | R > as a sympy expression.
        # L, R are instances of SlaterDet

        if not R.is_compatible(L):
            return _SP_ZERO

        if self.precalculated_half_dets and op in ('H', 'S'):
            # Fast path is only valid when both dets have the same alpha / beta
            # electron counts as the precomputed basis. Two-electron operator
            # construction synthesises sub-determinants with fewer electrons,
            # which don't live in the precomputed lookup tables -- fall through
            # to the general Hartree-product expansion below.
            iLa = self.lookup_a.get(L.alpha_string)
            iRa = self.lookup_a.get(R.alpha_string)
            iLb = self.lookup_b.get(L.beta_string.lower())
            iRb = self.lookup_b.get(R.beta_string.lower())
            if None not in (iLa, iRa, iLb, iRb):
                if op == 'H':
                    return self.aH[iLa, iRa] * self.bS[iLb, iRb] + self.aS[iLa, iRa] * self.bH[iLb, iRb]
                return self.aS[iLa, iRa] * self.bS[iLb, iRb]

        R_orbs, R_signs = R.get_orbital_permutations()
        terms = []
        for R_orb, R_sign in zip(R_orbs, R_signs):
            elem = self.op_orbprod(L.det_string, R_orb, op=op)
            if elem is _SP_ZERO:
                continue
            terms.append(elem if R_sign == 1 else -elem)

        if not terms:
            return _SP_ZERO
        if len(terms) == 1:
            return terms[0]
        return sp.Add(*terms)

    def op_fixed_psi(self, L, R, op='H'):
        if len(L) == 0:
            return _SP_ONE if op == 'S' else _SP_ZERO

        sum_terms = []
        for detL, cL in L:
            for detR, cR in R:
                elem = self.op_det(detL, detR, op=op)
                if elem is _SP_ZERO:
                    continue
                prd = cL * cR
                if prd == 1:
                    sum_terms.append(elem)
                elif prd == -1:
                    sum_terms.append(-elem)
                else:
                    sum_terms.append(prd * elem)

        if not sum_terms:
            return _SP_ZERO
        if len(sum_terms) == 1:
            return sum_terms[0]
        return sp.Add(*sum_terms)

    def Op(self, L, R, op='H'):
        L = FixedPsi(L)
        R = FixedPsi(R)
        return self.op_fixed_psi(L, R, op=op)

    def Ops(self, L, R, op='H', find_factors=True):
        z = self.Op(L=L, R=R, op=op)
        if find_factors:
            z = sp.factor(z)
        return z

    def getS(self, L, R, find_factors=True):
        return self.Ops(L, R, op='S', find_factors=find_factors)

    def getH(self, L, R, find_factors=True):
        return self.Ops(L, R, op='H', find_factors=find_factors)

    def build_matrix(self, u, op='H'):
        """
        Builds a square matrix of integrals for each pair of wavefunctions in a given array
        :param u: array of FixedPsi, SlaterDet, or str
        :param op: the integration operator
        :return: SymPy matrix with integrals
        """
        N = len(u)
        m = sp.zeros(N)
        if N == 0:
            return m

        # Coerce each entry to a FixedPsi for uniform handling, and pre-compute
        # the set of spin patterns it carries. Two FixedPsi are guaranteed to
        # yield zero whenever their spin-pattern sets are disjoint -- skip those
        # pairs entirely instead of running the full Op machinery.
        psis = [None] * N
        spin_keys = [None] * N
        for i, entry in enumerate(u):
            if isinstance(entry, str):
                fp = FixedPsi(entry)
            elif entry.__class__.__name__ == 'SlaterDet':
                fp = FixedPsi(entry)
            else:
                fp = entry
            psis[i] = fp
            spin_keys[i] = frozenset(d.spins for d in fp.dets)

        # Inline fast-path: when half-dets are precomputed and every basis entry
        # is a single Slater determinant with unit coefficient, skip the
        # FixedPsi/Op layer and index aH/aS/bH/bS directly.
        fast = (self.precalculated_half_dets and op in ('H', 'S')
                and all(len(p.dets) == 1 and p.coefs[0] == 1 for p in psis))

        if fast:
            half = [None] * N
            for i, p in enumerate(psis):
                d = p.dets[0]
                half[i] = (self.lookup_a[d.alpha_string],
                           self.lookup_b[d.beta_string.lower()],
                           d.spins)
            aH, aS, bH, bS = self.aH, self.aS, self.bH, self.bS
            for i in range(N):
                iLa, iLb, sL = half[i]
                for j in range(i, N):
                    iRa, iRb, sR = half[j]
                    if sL != sR:
                        continue
                    if op == 'H':
                        v = aH[iLa, iRa] * bS[iLb, iRb] + aS[iLa, iRa] * bH[iLb, iRb]
                    else:
                        v = aS[iLa, iRa] * bS[iLb, iRb]
                    m[i, j] = v
                    if i != j:
                        m[j, i] = v
            return m

        for i in range(N):
            ki = spin_keys[i]
            for j in range(i, N):
                if ki.isdisjoint(spin_keys[j]):
                    continue
                v = self.op_fixed_psi(psis[i], psis[j], op=op)
                m[i, j] = v
                if i != j:
                    m[j, i] = v
        return m

    def energy(self, P, o2=False):
        """
        Find the energy for the FixedPsi object: E = <P|H|P> / <P|P>
        :param P: A wavefunction: FixedPsi, SlaterDet, or str
        :return: Expression for the normalized energy: N_el * <P | H | P> / <P | P>
        """
        E = self.Ops(P, P, op='H')
        S = self.Ops(P, P, op='S')
        if o2:
            return (E / S) + sp.simplify(sp.simplify(self.o2_fixed_psi(P, P)) / S)
        else:
            return E / S

    def couple(self, P=None, mS=None, mH=None, N_tries=10, precision=12, ranges={'h':(-1.0,0.0),'s':(0.0,1.0)}, nums=None):
        """
        Group the FixedPsi objects that have constant ratios in the lowest energy wave vector
        The constant ratios are found by numerical simulation
        :param P: list of FixedPsi objects
        :param N_tries: number of trials
        :param precision: 10^-precision is the matching threshold
        :return:
        """
        if mS is None:
            mS = self.build_matrix(P, op='S')
        if mH is None:
            mH = self.build_matrix(P, op='H')

        ranges2 = {}
        symbols = mH.free_symbols.union(mS.free_symbols)
        for s in symbols:
            ss = str(s)
            if ss in ranges:
                ranges2[s] = ranges[ss]
            else:
                assert nums is not None and ss in nums, "Missing numerical value for the parameter " + ss
                ranges2[s] = (nums[ss], nums[ss])

        couplings = get_coupled(mS=mS, mH=mH, N_tries=N_tries, precision=precision, ranges=ranges2)
        return get_combined_from_dict(P, couplings)

    def get_o2_name(self, v):
        """
        Gets the standardized name of the 2e integral. Uses integral symmetries to sort indices.
        Substitutes the integral by name if provided
        Parameters
        ----------
        v: list of one-letter lower-case orbital names

        Returns
        -------
        string with the integral name
        """
        if len(numpy.unique((v[0].lower(), v[1].lower(), v[2].lower(), v[3].lower()))) > self.max_2e_centers:
            return '0'
        tiv = tuple(sort_ind(v))
        indices = '%s%s%s%s' % tiv
        int_name = 'T_%s' % indices

        if self.subst_2e is not None:
            r = '%s%s%s%s' % tuple(rankdata(tiv, method='dense'))
            if r in self.subst_2e:
                int_name = self.subst_2e[r]
        return int_name

    def o2_det(self, D1, D2):
        """
        Computes the two-electron integrals between two determinants
        Parameters
        ----------
        D1, D2: two objects SlaterDet

        Returns
        -------
        Sympy symbolic expression, T_abcd \equiv <ab|cd>
        """
        assert D1.Nel == D2.Nel, 'Different number of electrons'
        Nel = D1.Nel
        D1s = D1.det_string
        D2s = D2.det_string
        off = int(Nel * (Nel - 1) / 2)
        result = ['', ] * 2 * off ** 2
        ind = 0
        for i in range(Nel):
            for j in range(i + 1, Nel):
                s1 = D1s[:i] + D1s[i + 1:j] + D1s[j + 1:]
                c1, c2 = D1s[i], D1s[j]
                sumL = c1.islower() + c2.islower()

                sd1, f1 = standardize_det(s1)

                for k in range(Nel):
                    for m in range(k + 1, Nel):
                        s2 = D2s[:k] + D2s[k + 1:m] + D2s[m + 1:]
                        c3, c4 = D2s[k], D2s[m]

                        if len(numpy.unique((c1.lower(),c2.lower(),c3.lower(),c4.lower()))) > self.max_2e_centers:
                            continue

                        sumR = c3.islower() + c4.islower()
                        if sumL != sumR:
                            continue

                        sd2, f2 = standardize_det(s2)

                        opS = self.Op(sd1, sd2, op='S')
                        if opS == '(0)':
                            continue

                        parity = (i + j + k + m + f1 + f2) % 2

                        if c1.islower() == c3.islower():
                            iv = (c1.lower(), c2.lower(), c3.lower(), c4.lower())
                            int_name = self.get_o2_name(iv)
                            sign = '(1)' if parity == 0 else '(-1)'
                            result[ind] = '%s * %s * (%s)' % (sign, int_name, opS)
                            ind += 1

                        if c1.islower() == c4.islower():
                            iv = (c1.lower(), c2.lower(), c4.lower(), c3.lower())
                            int_name = self.get_o2_name(iv)
                            sign = '(1)' if parity == 1 else '(-1)'
                            result[ind] = '%s * %s * (%s)' % (sign, int_name, opS)
                            ind += 1

        if ind==0:
            return '0'
        else:
            return ' + '.join(result[:ind])

    def o2_fixed_psi(self, L, R, op='H'):

        if len(L) == 0:
            return 0

        vo = ['', ] * 2 * len(L)
        io = 0
        for detL, cL in L:
            vi = ['', ] * 2 * len(R)
            ii = 0
            for detR, cR in R:
                elem = self.o2_det(detL, detR)

                prd = attempt_int(cL * cR)
                if prd == 1:
                    prefix = '+'
                elif prd == -1:
                    prefix = '-'
                else:
                    prefix = '+(%s)*' % str(prd)

                vi[ii] = '%s(%s)' % (prefix, elem)
                ii += 1

            vo[io] = '(%s)' % ''.join(vi[:ii])
            io += 1

        s = '+'.join(vo[:io])
        # simple cleanup
        if s[0] == '+':
            s = s[1:]
        return s

    def o2(self, L, R, op='H'):
        L = FixedPsi(L)
        R = FixedPsi(R)
        return self.o2_fixed_psi(L, R)

    def o2_matrix(self, u):
        Nd = len(u)
        o2 = sp.zeros(Nd)
        for i in range(Nd):
            for j in range(i, Nd):
                o2[i, j] = self.o2(u[i], u[j])
                if i != j:
                    o2[j, i] = o2[i, j]
        return o2

    def o2_mo2ao(self, c1, c2, c3, c4):
        s = '0'
        for i1, k1 in c1:
            for i2, k2 in c2:
                for i3, k3 in c3:
                    for i4, k4 in c4:
                        if i1.det_string.isupper() != i3.det_string.isupper():
                            continue
                        if i2.det_string.isupper() != i4.det_string.isupper():
                            continue
                        s += ' + ' + str(k1 * k2 * k3 * k4) + '*' + self.get_o2_name((i1.det_string.lower(),
                                                                                      i2.det_string.lower(),
                                                                                      i3.det_string.lower(),
                                                                                      i4.det_string.lower()))
        return s

    def get_mo_norm(self, mo):
        # returns a diagonal matrix with the normalization factors on the diagonal
        mo_norm = sp.zeros(len(mo))
        for i in range(len(mo)):
            mo_norm[i, i] = 1 / sp.sqrt(self.Ops(mo[i], mo[i], op='S'))
        return mo_norm

    def get_fock(self, mo, Nel):
        # example of mo: [|a|+|b|, |A|+|B|, |a|-|b|, |A|-|B|]
        # Construct the Fock matrix
        Nmo = len(mo)
        fock = sp.zeros(Nmo)
        mo_norm = self.get_mo_norm(mo)
        for i in range(Nmo):
            for j in range(Nmo):
                for b in range(Nel):
                    norm = mo_norm[i,i] * mo_norm[j,j] * mo_norm[b,b]**2

                    fock[i,j] += sp.simplify(self.o2_mo2ao(mo[i],mo[b],mo[j],mo[b])) * norm
                    fock[i,j] -= sp.simplify(self.o2_mo2ao(mo[i],mo[b],mo[b],mo[j])) * norm

                fock[i,j] = sp.simplify(fock[i,j])

        return fock

    def get_rhf_fock(self, mo, Nel):
        # example of mo: [|a|+|b|, |A|+|B|, |a|-|b|, |A|-|B|]
        # Construct the Fock matrix
        Nmo = len(mo)
        fock = sp.zeros(Nmo)
        mo_norm = self.get_mo_norm(mo)
        for mu in range(Nmo):
            for nu in range(Nmo):
                for a in range(Nel // 2):
                    norm = mo_norm[mu, mu] * mo_norm[nu, nu] * mo_norm[a, a]**2

                    fock[mu, nu] += sp.simplify(self.o2_mo2ao(mo[mu], mo[a], mo[nu], mo[a])) * 2 * norm
                    fock[mu, nu] -= sp.simplify(self.o2_mo2ao(mo[mu], mo[a], mo[a], mo[nu])) * norm

                fock[mu, nu] = sp.simplify(fock[mu, nu])

        return fock

    def get_rhf_mo_energies(self, mo_rhf, Nel):
        mo_rhf_norm = self.get_mo_norm(mo_rhf)
        rhf_o1 = simplify_matrix(mo_rhf_norm * self.build_matrix(mo_rhf, op='H') * mo_rhf_norm)
        rhf_fock = self.get_rhf_fock(mo_rhf, Nel=Nel)
        return (rhf_o1 + rhf_fock).diagonal()


