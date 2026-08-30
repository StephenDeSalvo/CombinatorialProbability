"""Correctness tests for the integer partition samplers.

The properties worth pinning down are:

  * every sample is a partition of the requested weight,
  * a part never exceeds the requested cap on part size,
  * the table method's unranking is a bijection onto the partitions it claims to
    enumerate -- checked exhaustively for small n, which is stronger than any
    amount of random sampling, and
  * the samplers are actually uniform, which correct sums alone do not show.

Run with:  nix-shell --run pytest
"""

import collections
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from CombinatorialProbability import IntegerPartition
from CombinatorialProbability.integer_partition import _sampling

# Kept so the unranking tests can pin the variate to a chosen rank and put the
# real generator back afterwards.
_RANDINT = _sampling.randint


# ---------------------------------------------------------------- helpers

def weight_of(partition):
    """Sum of a partition held as {part_size: multiplicity}."""
    return sum(part * mult for part, mult in partition.items())


def all_partitions(n, max_part=None):
    """Every partition of n as a sorted descending tuple, for exhaustive checks."""
    if max_part is None:
        max_part = n
    if n == 0:
        return [()]
    out = []
    for part in range(min(n, max_part), 0, -1):
        for rest in all_partitions(n - part, part):
            out.append((part,) + rest)
    return out


def as_tuple(partition):
    """{3: 2, 1: 1} -> (3, 3, 1), so samples can be compared with all_partitions."""
    parts = []
    for part in sorted(partition, reverse=True):
        parts.extend([part] * partition[part])
    return tuple(parts)


def fitted(n):
    ip = IntegerPartition()
    ip.fit(weight=n, make_array=True, make_table=True, make_tilt=True)
    return ip


def draw(ip, method, size, **method_params):
    kwargs = {'size': size, 'method': method}
    if method_params:
        kwargs['method_params'] = method_params
    return ip.sampling(**kwargs)[0]


METHODS = ['rejection', 'pdcdsh', 'table_only', 'array_only']


# ---------------------------------------------------------------- validity

@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('n', [1, 2, 5, 10, 25])
def test_samples_are_partitions_of_n(method, n):
    """Every sample sums to n, with positive parts and positive multiplicities."""
    for partition in draw(fitted(n), method, 60):
        assert weight_of(partition) == n, f'{method} produced {partition}, weight {weight_of(partition)} != {n}'
        assert all(part >= 1 for part in partition), f'{method} produced a non-positive part: {partition}'
        assert all(mult >= 1 for mult in partition.values()), f'{method} produced a non-positive multiplicity: {partition}'


@pytest.mark.parametrize('n', [10, 25, 40])
@pytest.mark.parametrize('rows', [1, 2, 3, 5])
def test_pdc_recursive_is_a_partition_of_n(n, rows):
    """PDC-recursive splices a geometric half onto a table-method half.

    It used to merge the two with dict.update(), so an oversized part from the
    table half overwrote the geometric multiplicity at that key and the sample
    came out light -- weights like 13 or 18 for a target of 25.
    """
    for partition in draw(fitted(n), 'pdc-recursive', 40, rows=rows):
        assert weight_of(partition) == n, f'rows={rows} produced {partition}, weight {weight_of(partition)} != {n}'


@pytest.mark.parametrize('target,rows', [(4, 2), (4, 3), (4, 5), (7, 2), (12, 3), (20, 3), (30, 5)])
def test_table_method_respects_the_part_size_cap(target, rows):
    """A part must never exceed `rows`, nor the weight left to spend.

    The decode used to add 1 to a search that already returned the right index
    whenever the variate landed exactly on a column value, so it emitted a part
    one larger than allowed -- at target=4, rows=2 that was a quarter of samples.
    """
    ip = fitted(max(target, 40))
    samples = _sampling.table_method_sampling(
        ip, table=ip.p_n_k_table, target=target, rows=rows, size=400)[0]
    for partition in samples:
        assert weight_of(partition) == target, f'{partition} has weight {weight_of(partition)}, wanted {target}'
        assert max(partition) <= rows, f'{partition} has a part above the cap rows={rows}'
        assert max(partition) <= target, f'{partition} has a part above the weight {target}'


# ---------------------------------------------------------------- unranking

@pytest.mark.parametrize('n', [1, 2, 3, 6, 9, 12])
def test_table_unranking_is_a_bijection(n):
    """Ranks 1..p(n) must decode to each partition of n exactly once.

    Exhaustive, so it settles both the ordering and the boundaries -- no
    sampling run can prove this.
    """
    ip = fitted(n)
    table = ip.p_n_k_table
    seen = []
    for rank in range(1, table[n][n] + 1):
        _sampling.randint = lambda a, b, r=rank: r        # pin the rank
        try:
            partition = _sampling.table_method_sampling(
                ip, table=table, target=n, size=1)[0][0]
        finally:
            _sampling.randint = _RANDINT
        seen.append(as_tuple(partition))

    expected = all_partitions(n)
    assert len(seen) == len(expected) == table[n][n]
    assert sorted(seen) == sorted(expected), 'unranking is not onto the partitions of n'
    assert len(set(seen)) == len(seen), 'unranking hit the same partition twice'


@pytest.mark.parametrize('n,rows', [(6, 2), (6, 3), (9, 3), (12, 4)])
def test_table_unranking_is_a_bijection_with_a_cap(n, rows):
    """Same, restricted to partitions with parts of size at most `rows`."""
    ip = fitted(n)
    table = ip.p_n_k_table
    seen = []
    for rank in range(1, table[rows][n] + 1):
        _sampling.randint = lambda a, b, r=rank: r
        try:
            partition = _sampling.table_method_sampling(
                ip, table=table, target=n, rows=rows, size=1)[0][0]
        finally:
            _sampling.randint = _RANDINT
        seen.append(as_tuple(partition))

    expected = all_partitions(n, max_part=rows)
    assert len(seen) == len(expected) == table[rows][n]
    assert sorted(seen) == sorted(expected)


def test_top_rank_is_reachable_and_valid():
    """Rank p(n) is a real partition, not one part too many.

    random.randint is inclusive at both ends, and the old code drew from
    0..p(n) -- so rank p(n) decoded to a part of size n+1, and rank 0 was not a
    partition at all. At n=10 that was about one sample in forty.
    """
    n = 10
    ip = fitted(n)
    table = ip.p_n_k_table
    for rank in (1, table[n][n]):
        _sampling.randint = lambda a, b, r=rank: r
        try:
            partition = _sampling.table_method_sampling(
                ip, table=table, target=n, size=1)[0][0]
        finally:
            _sampling.randint = _RANDINT
        assert weight_of(partition) == n
        assert max(partition) <= n


# ---------------------------------------------------------------- uniformity

@pytest.mark.parametrize('method', METHODS)
def test_sampler_is_uniform(method):
    """A chi-square goodness-of-fit against the uniform distribution on p(n).

    Summing to n does not make a sampler correct -- it has to hit every
    partition equally often. n=8 has 22 partitions, small enough to enumerate
    and to keep the expected cell counts comfortably above 5.
    """
    from scipy.stats import chisquare

    n = 8
    expected_support = all_partitions(n)
    draws = 22 * 250

    counts = collections.Counter(as_tuple(p) for p in draw(fitted(n), method, draws))
    assert set(counts) <= set(expected_support), (
        f'{method} produced something that is not a partition of {n}: '
        f'{set(counts) - set(expected_support)}')

    observed = [counts.get(part, 0) for part in expected_support]
    _, p_value = chisquare(observed)
    assert p_value > 0.001, (
        f'{method} does not look uniform over the {len(expected_support)} '
        f'partitions of {n} (chi-square p={p_value:.2e})')


@pytest.mark.parametrize('rows', [2, 3])
def test_pdc_recursive_is_uniform(rows):
    """Same check for the hybrid, which has the most moving parts."""
    from scipy.stats import chisquare

    n = 8
    expected_support = all_partitions(n)
    counts = collections.Counter(
        as_tuple(p) for p in draw(fitted(n), 'pdc-recursive', 22 * 250, rows=rows))
    assert set(counts) <= set(expected_support), (
        f'rows={rows} produced a non-partition: {set(counts) - set(expected_support)}')

    observed = [counts.get(part, 0) for part in expected_support]
    _, p_value = chisquare(observed)
    assert p_value > 0.001, f'pdc-recursive rows={rows} not uniform (p={p_value:.2e})'
