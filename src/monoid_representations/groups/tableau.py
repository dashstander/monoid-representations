from copy import deepcopy
from functools import cache, total_ordering


def _subpartitions(part):
    assert len(part) >= 2
    new_parts = []
    if part[0] == 1:
        yield part
    part = list(part)
    for i in range(len(part) - 1):
        currval = part[i]
        nextval = part[i+1]
        if (currval - nextval) > 1:
            new_part = deepcopy(part)
            new_part[i] = currval - 1
            new_part[i + 1] = nextval + 1
            new_parts.append(tuple(new_part))
        if currval > 1 and nextval == 1:
            new_part = deepcopy(part)
            new_part[i] = currval - 1
            new_part.append(1)
            new_parts.append(tuple(new_part))
    if part[-1] > 1:
        new_part = deepcopy(part)
        lastval = new_part[-1]
        new_part[-1] = lastval - 1
        new_part.append(1)
        new_parts.append(tuple(new_part))
    for subpart in new_parts:
        yield subpart
        for subsub in _subpartitions(subpart):
            yield subsub


@cache
def _generate_partitions(n):
    if n == 3:
        partitions = [(3,), (2, 1), (1, 1, 1)]
    elif n == 2:
        partitions = [(2,), (1, 1)]
    elif n == 1:
        partitions = [(1,)]
    elif n == 0:
        return ()
    else:
        partitions = [(n,)]
        for k in range(n):
            m = n - k
            partitions.extend(
                tuple(
                    sorted((m, *p), reverse=True)
                ) for p in _generate_partitions(k)
            )
    return partitions


def partitions_beneath(partition):
    n = sum(partition)
    results = {partition: set()}

    def generate_partitions(lattice_partition):
        num_pieces = len(lattice_partition)
        
        if lattice_partition not in results:
            results[lattice_partition] = set()
            
        if num_pieces == n:
            results[lattice_partition].add(lattice_partition[1:])
            return

        current_partition = list(lattice_partition)
        current_partition.append(0)
        
        for i, curr_val in enumerate(current_partition):
            diffs = [curr_val - other for other in current_partition[i+1:]]

            if len(diffs) == 0:
                break
                
            if len(diffs) == 1 and curr_val == 1:
                new_partition = deepcopy(current_partition[:-2])
                results[lattice_partition].add(tuple(new_partition))
                continue    
                    
            #print(current_partition, curr_val, diffs)

            if diffs[0] == 0:
                continue

            
            for j, diff in enumerate(diffs):
                pos = i + j + 1
                val = current_partition[pos]
                #assert (curr_val - val == diff), val
                if (curr_val - val > 1) and (current_partition[pos-1] - val > 0):
                    new_partition = deepcopy(current_partition)
                    new_partition[i] -= 1
                    new_partition[pos] += 1
                    
                    if new_partition[-1] == 0:
                        new_partition.pop(-1)

                    if len(new_partition) > 1:
                        generate_partitions(tuple(new_partition))

                    if new_partition[-1] == 1:
                        results[lattice_partition].add(tuple(new_partition[:-1]))
                    elif new_partition[-1] > 1:
                        new_partition[-1] -= 1
                        results[lattice_partition].add(tuple(new_partition))
                    else:
                        raise ValueError('We never should have reached this spot')

    generate_partitions(partition)
    return {k: list(v) for k, v in results.items()}


def check_parity(partition):
    even_cycles = [c for c in partition if (c % 2 == 0)]
    return len(even_cycles) % 2


def generate_partitions(n):
    return sorted(list(set(_generate_partitions(n))))


def conjugate_partition(partition):
    n = sum(partition)
    conj_part = []
    for i in range(n):
        reverse = [j for j in partition if j > i]
        if reverse:
            conj_part.append(len(reverse))
    return tuple(conj_part)
    

@total_ordering
class YoungTableau:

    def __init__(self, values: list[list[int]]):
        self.values = values
        self.shape = tuple([len(row) for row in values])
        self.n = sum(self.shape)

    def __repr__(self):
        strrep = []
        for row in self.values:
            strrep.append('|' + '|'.join([str(v) for v in row]) + '|' )
        return '\n'.join(strrep)

    def __len__(self):
        return self.n
    
    def __getitem__(self, key):
        i, j = key
        return self.values[i][j]
    
    def __setitem__(self, key, value):
        i, j = key
        self.values[i][j] = value

    def __eq__(self, other):
        if not isinstance(other, YoungTableau):
            other = YoungTableau(other)
        if (self.n != other.n) or (self.shape != other.shape):
            return False
        for row1, row2 in zip(self.values, other.values):
            if row1 != row2:
                return False
        return True

    def __lt__(self, other):
        if not isinstance(other, YoungTableau):
            other = YoungTableau(other)
        if (self.n != other.n) or (self.shape != other.shape):
            raise ValueError('Can only compare two tableau of the same shape')
        for row1, row2 in zip(self.values, other.values):
            if row1 == row2:
                continue
            for v1, v2 in zip(row1, row2):
                if v1 == v2:
                    continue
                return v1 < v2

    def index(self, val):
        for r, row in enumerate(self.values):
            if val in row:
                c = row.index(val)
                return r, c
        raise ValueError(f'{val} could not be found in tableau.')
    
    def transposition_dist(self, x: int, y: int) -> int:
        #assert y == x + 1
        row_x, col_x = self.index(x)
        row_y, col_y = self.index(y)
        row_dist = row_x - row_y
        col_dist = col_y - col_x
        return row_dist + col_dist


def generate_standard_young_tableaux(lambda_partition):
    n = sum(lambda_partition)
    tableau = [[None] * x for x in lambda_partition]
    tableau[0][0] = 1
    
    def fill_tableau(tab, num):
        if num == n + 1:
            yield tuple([tuple(row) for row in tab])
        else:
            for i, row in enumerate(tab):
                for j, val in enumerate(row):
                    if val is None:
                        #print(val, i, j)
                        above_smaller = (i == 0) or ((tab[i-1][j] is not None) and (tab[i-1][j] < num))
                        right_smaller = (j == 0) or ((tab[i][j-1] is not None) and (tab[i][j-1] < num))
                        if above_smaller and right_smaller:
                            new_tab = deepcopy(tab)
                            new_tab[i][j] = num
                            yield from fill_tableau(new_tab, num + 1)

    yield from fill_tableau(tableau, 2)
