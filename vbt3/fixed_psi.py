import vbt3
from vbt3.functions import generate_det_strings, standardize_det, standardize_det_2
from vbt3.functions import attempt_int
import copy


class FixedPsi:
    # Linear combination of determinants
    def __init__(self, x=None, coupled_pairs=None):
        """

        """
        self.Nel = 0

        self.dets = []
        self.coefs = []

        if x is None:
            return
        self.__iadd__(x)

        if coupled_pairs is not None:
            for i, j in coupled_pairs:
                self.couple_orbitals(i, j)

    def __iadd__(self, other):
        if other.__class__.__name__ == 'str':
            self.add_str_det(other)
        elif other.__class__.__name__ == 'SlaterDet':
            self.add_det(other)
        elif other.__class__.__name__ == 'FixedPsi':
            self.add_fixedpsi(other)
        return self

    def __add__(self, other):
        result = FixedPsi(self)
        result += other
        return result

    def __sub__(self, other):
        result = FixedPsi(self)
        result += (-1)*other
        return result

    def __rsub__(self, other):
        result = (-1) * FixedPsi(self)
        result += other
        return result

    def __neg__(self):
        return (-1) * FixedPsi(self)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            result = FixedPsi(self)
            for i in range(len(result)):
                result.coefs[i] = attempt_int(result.coefs[i]*other)
            return result

        if other.__class__.__name__ == 'SlaterDet':
            result = FixedPsi()
            for d, c in self:
                result.add_det(d * other, c)
            return result

        if other.__class__.__name__ == 'FixedPsi':
            result = FixedPsi()
            for dS, cS in self:
                for dO, cO in other:
                    d = dS * dO
                    if d.Nel == 0:
                        continue
                    fp1 = d.get_sorted()
                    c1, d1 = fp1.coefs[0], fp1.dets[0]
                    fp2 = standardize_det_2(d1)
                    result = result + (c1*cS*cO)*fp2
                    # result.add_det(d, coef = cS * cO)
            return result
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            result = FixedPsi(self)
            for i in range(len(result)):
                result.coefs[i] = attempt_int(result.coefs[i]*other)
            return result

    def __getitem__(self, item):
        return self.dets[item]

    def __len__(self):
        return len(self.dets)

    def __contains__(self, item):
        for d, c in self:
            if d.det_string == item:
                return True
        return False

    def __iter__(self):
        return zip(self.dets, self.coefs)

    def add_det(self, det, coef=+1):
        assert det.__class__.__name__ == 'SlaterDet'
        if det.Nel == 0:
            return

        assert self.Nel in (0, det.Nel)

        for i in range(len(self)):
            if det.det_string == self.dets[i].det_string:
                self.coefs[i] += coef
                return

        self.dets.append(det)
        cf = attempt_int(coef)
        self.coefs.append(cf)
        if self.Nel == 0:
            self.Nel = det.Nel

    def add_str_det(self, det_string, coef=+1):
        sd = vbt3.SlaterDet(det_string)
        self.add_det(sd, coef=coef)

    def add_fixedpsi(self, p, coef=1.0):
        for d, c in p:
            merged = False
            # check if d is already in self
            for i in range(len(self)):
                if self.dets[i].det_string == d.det_string:
                    self.coefs[i] += c
                    # if it turns to 0, shift left the remaining dets
                    if self.coefs[i] == 0:
                        for j in range(i, len(self) - 1):
                            self.dets[j] = self.dets[j + 1]
                            self.coefs[j] = self.coefs[j + 1]
                        self.dets = self.dets[:-1]
                        self.coefs = self.coefs[:-1]
                    merged = True
                    break
            if not merged:
                self.add_det(d, c * coef)

    def couple_orbitals(self, o1, o2):
        # generate determinants that represent a singlet bonding coupling between two orbitals.
        # Orbital numbering starts from 0
        determinants = self.dets.copy()
        coefs = self.coefs.copy()
        for d, coef in zip(determinants, coefs):
            # Flip spins
            ds = d.det_string
            c1, c2 = [c.lower() if c.isupper() else c.upper() for c in [ds[o1], ds[o2]]]
            assert c1.lower() != c2.lower(), 'Cannot couple the same orbital'
            # Flip positions
            ds2 = ds[:o1] + c2 + ds[(o1 + 1):o2] + c1 + ds[(o2 + 1):]
            self.add_str_det(ds2, coef=coef)

    def __repr__(self):
        s = ''
        for d, cf in self:
            dc = attempt_int(cf)

            if dc > 0:
                if dc == 1.0:
                    s += '+'
                else:
                    s += '+%s' % dc
            elif dc < 0:
                if dc == -1.0:
                    s += '-'
                else:
                    s += '%s' % dc
            s += str(d)
        if len(s) > 0 and s[0] == '+':
            s = s[1:]
        return s


def generate_dets(Nela, Nelb, Norb):
    """
    Generate all possible determinants for a given number of electrons and atomic orbitals.
    :param Nela: Number of alpha electrons
    :param Nelb: Number of beta electrons
    :param Norb: Number of atomic orbitals
    :return: List of FixedPsi objects, each containing one determinant
    """
    L = generate_det_strings(Nela, Nelb, Norb)
    PP = [None,]*len(L)
    for i in range(len(L)):
        PP[i] = FixedPsi(L[i])
    return PP
