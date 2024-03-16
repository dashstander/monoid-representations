from functools import reduce
from itertools import combinations, pairwise
import numpy as np
import torch

from .permutations import Permutation
from .tableau import generate_standard_young_tableaux


def adj_trans_decomp(i: int, j: int) -> list[tuple[int]]:
    center = [(i, i + 1)]
    i_to_j = list(range(i+1, j+1))
    adj_to_j = list(pairwise(i_to_j))
    return list(reversed(adj_to_j)) + center + adj_to_j


def cycle_to_one_line(cycle_rep):
    n = sum([len(c) for c in cycle_rep])
    sigma = [-1] * n
    for cycle in cycle_rep:
        first = cycle[0]
        if len(cycle) == 1:
            sigma[first] = first
        else:
            for val1, val2 in pairwise(cycle):
                sigma[val2] = val1
                lastval  = val2
            sigma[first] = lastval
    return tuple(sigma)


def trans_to_one_line(i, j, n):
    sigma = list(range(n))
    sigma[i] = j
    sigma[j] = i
    return tuple(sigma)


def adjacent_transpositions(n):
    return pairwise(range(n))
    

def non_adjacent_transpositions(n):
    return [(i, j) for i, j in combinations(range(n), 2) if i+1 != j]


def adjacent_transposition_matrix(tableaux_basis, a, b):
    dim = len(tableaux_basis)
    n = tableaux_basis[0].n
    perm = Permutation.transposition(n, a, b)
    irrep = np.zeros((dim, dim))
    def fn(i, j):
        tableau = tableaux_basis[i]
        if i == j:
            d = tableau.transposition_dist(a, b)
            return 1. / d
        else:
            new_tab = perm * tableau
            if new_tab == tableaux_basis[j]:
                d = tableau.transposition_dist(a, b)**2
                return np.sqrt(1 - (1. / d))
            else:
                return 0.
    for x in range(dim):
        for y in range(dim):
            irrep[x, y] = fn(x, y)
    return irrep


def generate_transposition_matrices(tableaux_basis):
    n = tableaux_basis[0].n
    matrices = {
        (i, j): adjacent_transposition_matrix(tableaux_basis, i, j) for i, j in adjacent_transpositions(n)
    }
    for i, j in non_adjacent_transpositions(n):
        decomp = [matrices[pair] for pair in adj_trans_decomp(i, j)]
        matrices[(i, j)] = reduce(lambda x, y: x @ y, decomp)
    return matrices


def matrix_representations(n: int, partition: tuple[int]):
    Sn = Permutation.full_group(n)
    basis = sorted(generate_standard_young_tableaux(partition))
    dim = len(basis)
    transpo_matrices = generate_transposition_matrices(basis)
    matrices = {
        trans_to_one_line(*k, n): v for k, v in transpo_matrices.items()
    }
    matrices[tuple(range(n))] = np.eye(dim)
    for perm in Sn:
        if perm.sigma in matrices:
            continue
        else:
            cycle_mats = [
                transpo_matrices[t] for t in perm.transposition_decomposition()
            ]
            perm_rep = reduce(lambda x, y: x @ y, cycle_mats)
            matrices[perm.sigma] = perm_rep
            if perm.inverse.sigma not in matrices:
                matrices[perm.inverse.sigma] = perm_rep.T
    return matrices


def irrep_tensor(representations, n):
    order = len(representations.keys())
    shape = representations[tuple(range(n))].shape
    tensor = torch.empty((order, *shape))
    for permutation, matrix in representations.items():
        idx = Permutation(permutation).index_of_n()
        tensor[idx] = matrix
    return tensor


class SnIrrep:

    def __init__(self, n: int, partition: tuple[int], representations):
        self.n = n
        self.partition = partition
        self.basis = sorted(generate_standard_young_tableaux(partition))
        self.dim = len(self.basis)
        self.irreps = representations
        self.permutations = Permutation.full_group(n)
        self.index = {perm.sigma: i for i, perm in enumerate(self.permutations)}

    @classmethod
    def generate_representations(cls, n: int, partition: tuple[int]):
        matrices = matrix_representations(n, partition)
        return cls(n, partition, irrep_tensor(matrices))
    
    def alternating_matrix_tensor(self):
        matrices = self.matrix_representations()
        tensors = [torch.asarray(matrices[perm.sigma]).unsqueeze(0) for perm in self.permutations if perm.parity == 0]
        return torch.concatenate(tensors, dim=0).squeeze()
