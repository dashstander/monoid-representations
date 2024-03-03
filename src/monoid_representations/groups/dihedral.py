from copy import deepcopy
from functools import reduce
from itertools import product
import math
import numpy as np
from operator import mul
import torch


class DihedralElement:

    def __init__(self, rot: int, ref: int, n: int):
        self.n = n
        self.rot = rot % n
        self.ref = ref % 2

    def __repr__(self):
        return str((self.rot, self.ref))
    
    def __hash__(self):
        return hash(str(self))
    
    @property
    def sigma(self):
        return self.rot, self.ref
    
    @classmethod
    def full_group(cls, n):
        return [
            DihedralElement(r, p, n) for r, p in product(range(n), [0, 1])
        ]
    
    @property
    def order(self):
        if self.ref:
            return 2
        elif self.rot == 0:
            return 0
        elif (self.n % self.rot) == 0:
            return self.n // self.rot
        else:
            return math.lcm(self.n, self.rot) // self.rot
        
    @property
    def inverse(self):
        if self.ref:
            return deepcopy(self)
        else:
            return DihedralElement(self.n - self.rot, self.ref, self.n)
    
    def __mul__(self, other):
        if (not isinstance(other, DihedralElement)) or (other.n != self.n):
            raise ValueError(
                'Can only multiply a dihedral rotation with another dihedral rotation'
            )
        if self.ref:
            rot = self.rot - other.rot
        else:
            rot = self.rot + other.rot
        return DihedralElement(rot, self.ref + other.ref, self.n)
    
    def __pow__(self, x: int):
        if x == -1:
            return self.inverse
        elif x == 0:
            return DihedralElement(0, 0, self.n)
        elif x == 1:
            return deepcopy(self)
        else:
            return reduce(mul, [deepcopy(self) for _ in range(x)])

    def index(self) -> int:
        return 2 * self.rot + self.ref


def generate_subgroup(generators, n):
    group_size = 0
    all_elements = set(generators)
    while group_size < len(all_elements):
        group_size = len(all_elements)
        rotations = [DihedralElement(*p, n) for p in all_elements]
        for r1, r2 in product(rotations, repeat=2):
            r3 = r1 * r2
            all_elements.add(r3.sigma)
    return list(all_elements)


def dihedral_conjugacy_classes(n: int):

    conj_classes = [(0, 0), (0, 1)]
    if n % 2 == 1:
        m = (n - 1) // 2
    else:
        m = (n - 2) // 2 
        conj_classes += [((2, 0), (0, 1)), ((2, 1), (1, 1))]
    conj_classes += [(i, 0) for i in range(1, m+1)]
    return conj_classes


class DihedralIrrep:

    def __init__(self, n: int, conjugacy_class):
        self.n = n
        self.conjugacy_class = conjugacy_class
        self.group = DihedralElement.full_group(n)

    def _trivial_irrep(self):
        return {r.sigma: torch.ones((1,)) for r in self.group}
    
    def _reflection_irrep(self):
        return {r.sigma: (-1**r.ref) * torch.ones((1,)) for r in self.group}
    
    def _subgroup_irrep(self, sg):
        return {r.sigma: 1 if r.sigma in sg else -1 for r in self.group}
    
    def _matrix_irrep(self, k):
        mats = {}
        mats[(0, 0)] = torch.eye(2)
        ref = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float32)
        mats[(0, 1)] = ref
        two_pi_n = 2 * torch.pi / self.n
        for idx in range(1, self.n):
            sin = np.sin(two_pi_n * idx * k)
            cos = np.cos(two_pi_n * idx * k)
            m = torch.tensor([[cos, -1.0 * sin], [sin, cos]], dtype=torch.float32)
            mats[(idx, 0)] = m
            mats[(idx, 1)] = ref @ m
        return mats
    
    def matrix_representations(self):
        if self.conjugacy_class == (0, 0):
            return self._trivial_irrep()
        elif self.conjugacy_class == (0, 1):
            return self._reflection_irrep()
        elif self.conjugacy_class[1] == 0:
            return self._matrix_irrep(self.conjugacy_class[0])
        elif (
            isinstance(self.conjugacy_class[0], tuple) and 
            isinstance(self.conjugacy_class[1], tuple)
        ):
            subgroup = generate_subgroup(self.conjugacy_class, self.n)
            return self._subgroup_irrep(subgroup)
        else:
            raise ValueError(
                f'Somehow {self.conjugacy_class} is not a proper conjugacy class....'
            )


class ProductDihedralIrrep:

    def __init__(self, n: int, m: int, conjugacy_classes):
        
        assert len(conjugacy_classes) == m
        
        self.m = m
        self.n = n
        self.conjugacy_class = conjugacy_classes
        self.group = [element for element in product(DihedralElement.full_group(n), repeat=m)]
        self.irreps = [
            DihedralIrrep(n, conj_class).matrix_representations() for conj_class in conjugacy_classes
        ]

    def matrix_representations(self):

        reps = {}

        for elements in self.group:
            key = tuple(el.sigma for el in elements)
            matrices = [self.irreps[i][el] for i, el in enumerate(key)]
            reps[key] = reduce(torch.kron, matrices)
        return reps

