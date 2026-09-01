"""Correctness tests for the set partition samplers.

The properties worth pinning down are:

  * every sample is a partition of {1, ..., n} -- every element once, no empty block,
  * a restriction on the number of blocks is obeyed, whether it is a cap or an exact count,
  * the table method's unranking is a bijection onto the partitions it claims to
    enumerate -- checked exhaustively for small n, which is stronger than any amount of
    random sampling -- and it agrees with the order the iterator walks in,
  * the counting functions agree with each other and with the known Bell numbers, and
  * the samplers are actually uniform, which valid samples alone do not show.

Run with:  nix-shell --run pytest
"""

import collections
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from CombinatorialProbability import SetPartition
from CombinatorialProbability.set_partition import _sampling
from CombinatorialProbability.set_partition import _transforms

# Kept so the unranking tests can pin the variate to a chosen rank and put the
# real generator back afterwards.
_RANDINT = _sampling.randint

# OEIS A000110.
BELL = [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570]


# ---------------------------------------------------------------- helpers

def all_set_partitions(n, max_blocks=None):
    """Every set partition of [n], built independently of the library.

    Grown one element at a time: element i either joins one of the blocks already there or
    starts its own.  Blocks come out ordered by least element, which is the canonical form
    the library returns, so the two are directly comparable.
    """
    if n == 0:
        return [()]
    partitions = [((1,),)]
    for element in range(2, n+1):
        extended = []
        for partition in partitions:
            for index in range(len(partition)):
                blocks = [list(block) for block in partition]
                blocks[index].append(element)
                extended.append(tuple(tuple(block) for block in blocks))
            if max_blocks is None or len(partition) < max_blocks:
                extended.append(partition + ((element,),))
        partitions = extended
    if max_blocks is not None and n >= 1 and max_blocks < 1:
        return []
    return partitions


def elements_of(partition):
    """Every element of a set partition, with repeats kept so they can be caught."""
    return sorted(element for block in partition for element in block)


def fitted(n, **restrictions):
    generator = SetPartition()
    generator.fit(weight=n, make_array=True, make_table=True, make_tilt=True, **restrictions)
    return generator


def draw(generator, method, size, **method_params):
    kwargs = {'size': size, 'method': method}
    if method_params:
        kwargs['method_params'] = method_params
    return generator.sampling(**kwargs)[0]


METHODS = ['rejection', 'stam', 'table_only', 'array_only', 'stirling']

# The methods that cost nothing per sample once their table is built, so a uniformity test
# can afford a bigger support for them.
EXACT_METHODS = ['stam', 'table_only', 'array_only', 'stirling']


# ---------------------------------------------------------------- validity

@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('n', [1, 2, 5, 8, 15])
def test_samples_are_set_partitions_of_n(method, n):
    """Every sample holds each of 1..n exactly once, in nonempty blocks."""
    for partition in draw(fitted(n), method, 60):
        assert elements_of(partition) == list(range(1, n+1)), (
            f'{method} produced {partition}, which is not a partition of [{n}]')
        assert all(len(block) >= 1 for block in partition), (
            f'{method} produced an empty block: {partition}')


@pytest.mark.parametrize('method', METHODS)
def test_empty_ground_set(method):
    """[0] has exactly one partition, the empty one, and no method may choke on it."""
    for partition in draw(fitted(0), method, 5):
        assert partition == ()


@pytest.mark.parametrize('n,cap', [(4, 2), (4, 3), (4, 9), (6, 3), (10, 2), (12, 4)])
def test_table_method_respects_the_block_cap(n, cap):
    """A sample must never use more than max_blocks blocks.

    The cap is built into the table rather than checked while decoding: with a cap of K,
    the table simply never counts a string that opens block K+1, so a rank cannot decode
    to one.  That makes it worth testing that the two really do line up.
    """
    generator = fitted(n, max_blocks=cap)
    for partition in draw(generator, 'table_only', 300):
        assert elements_of(partition) == list(range(1, n+1))
        assert len(partition) <= cap, f'{partition} uses more than {cap} blocks'


@pytest.mark.parametrize('n,k', [(1, 1), (5, 2), (6, 3), (7, 1), (7, 7), (12, 4)])
def test_stirling_method_hits_the_exact_block_count(n, k):
    """blocks=k must give exactly k blocks, every time."""
    generator = fitted(n, blocks=k)
    for partition in draw(generator, 'stirling', 200):
        assert elements_of(partition) == list(range(1, n+1))
        assert len(partition) == k, f'{partition} has {len(partition)} blocks, wanted {k}'


@pytest.mark.parametrize('n,k', [(5, 0), (5, 6), (5, -1)])
def test_impossible_block_counts_are_refused(n, k):
    """There is no partition of [5] into 6 blocks, and saying so beats returning nonsense."""
    with pytest.raises(ValueError):
        draw(fitted(n, blocks=k), 'stirling', 1)


@pytest.mark.parametrize('method', ['rejection', 'stam', 'array_only'])
@pytest.mark.parametrize('restriction', [{'max_blocks': 2}, {'blocks': 2}])
def test_methods_refuse_restrictions_they_cannot_honour(method, restriction):
    """Only the table method can cap the blocks and only the Stirling method can fix them.

    The rest have to say so.  Ignoring the restriction would leave a sampler that is
    uniform over the wrong set -- every sample valid, every test of validity passing.
    """
    with pytest.raises(ValueError):
        draw(fitted(6, **restriction), method, 1)


def test_table_method_refuses_an_exact_block_count():
    with pytest.raises(ValueError):
        draw(fitted(6, blocks=3), 'table_only', 1)


def test_stirling_method_refuses_a_cap():
    with pytest.raises(ValueError):
        draw(fitted(6, max_blocks=3), 'stirling', 1)


# ---------------------------------------------------------------- unranking

@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 6])
def test_rgs_unranking_is_a_bijection(n):
    """Ranks 1..B(n) must decode to each set partition of [n] exactly once.

    Exhaustive, so it settles both the ordering and the boundaries -- no sampling run can
    prove this.
    """
    generator = fitted(n)
    table = generator.make_rgs_table(n)
    seen = []
    for rank in range(1, table[n][0] + 1):
        _sampling.randint = lambda a, b, r=rank: r        # pin the rank
        try:
            partition = _sampling.table_method_sampling(
                generator, table=table, target=n, size=1)[0][0]
        finally:
            _sampling.randint = _RANDINT
        seen.append(partition)

    expected = all_set_partitions(n)
    assert len(seen) == len(expected) == BELL[n]
    assert sorted(seen) == sorted(expected), 'unranking is not onto the set partitions of [n]'
    assert len(set(seen)) == len(seen), 'unranking hit the same partition twice'


@pytest.mark.parametrize('n,cap', [(4, 2), (5, 2), (5, 3), (6, 3), (6, 4)])
def test_rgs_unranking_is_a_bijection_with_a_cap(n, cap):
    """Same, restricted to partitions into at most `cap` blocks."""
    generator = fitted(n, max_blocks=cap)
    table = generator.make_rgs_table(n, cap)
    seen = []
    for rank in range(1, table[n][0] + 1):
        _sampling.randint = lambda a, b, r=rank: r
        try:
            partition = _sampling.table_method_sampling(
                generator, table=table, target=n, max_blocks=cap, size=1)[0][0]
        finally:
            _sampling.randint = _RANDINT
        seen.append(partition)

    expected = all_set_partitions(n, max_blocks=cap)
    assert len(seen) == len(expected) == table[n][0]
    assert sorted(seen) == sorted(expected)


def test_top_rank_is_reachable_and_valid():
    """Ranks 1 and B(n) are both real partitions.

    random.randint is inclusive at both ends, so a draw from 0..B(n) rather than 1..B(n)
    would put one rank past the end of the table and one before its start; neither decodes
    to anything.
    """
    n = 7
    generator = fitted(n)
    table = generator.make_rgs_table(n)
    for rank in (1, table[n][0]):
        _sampling.randint = lambda a, b, r=rank: r
        try:
            partition = _sampling.table_method_sampling(
                generator, table=table, target=n, size=1)[0][0]
        finally:
            _sampling.randint = _RANDINT
        assert elements_of(partition) == list(range(1, n+1))

    # And the extremes of the order are the two partitions one can name in advance.
    assert _decode(generator, table, n, 1) == (tuple(range(1, n+1)),)
    assert _decode(generator, table, n, table[n][0]) == tuple((i,) for i in range(1, n+1))


def _decode(generator, table, n, rank):
    _sampling.randint = lambda a, b, r=rank: r
    try:
        return _sampling.table_method_sampling(generator, table=table, target=n, size=1)[0][0]
    finally:
        _sampling.randint = _RANDINT


@pytest.mark.parametrize('n', [1, 4, 6])
def test_unranking_order_matches_the_iterator(n):
    """Rank r decodes to the r-th partition the iterator yields.

    Both are meant to be lexicographic on the restricted growth string, and they are
    written independently -- one walks a table, the other steps a string -- so agreeing is
    worth something.
    """
    generator = fitted(n)
    table = generator.make_rgs_table(n)
    walked = list(fitted(n))
    assert len(walked) == BELL[n]
    for rank, partition in enumerate(walked, start=1):
        assert _decode(generator, table, n, rank) == partition


# ---------------------------------------------------------------- counting

def test_bell_numbers():
    """B(0) .. B(11) against the known values, and the array extends rather than restarts."""
    generator = SetPartition()
    assert [generator.bell(n) for n in range(len(BELL))] == BELL
    # Asking again, and asking for less than is already there, must not disturb it.
    assert generator.bell(3) == 5
    assert generator.bell(11) == BELL[11]


def test_stirling_rows_sum_to_the_bell_numbers():
    """sum_k S(n,k) = B(n), which ties the 2D table to the 1D array."""
    generator = SetPartition()
    for n in range(len(BELL)):
        assert sum(generator.s_n_k(n, k) for k in range(n+1)) == BELL[n]


def test_stirling_edges():
    """The values one can write down: S(n,1) = S(n,n) = 1, S(n,n-1) = C(n,2), S(0,k) = 0."""
    generator = SetPartition()
    for n in range(1, 10):
        assert generator.s_n_k(n, 1) == 1
        assert generator.s_n_k(n, n) == 1
        assert generator.s_n_k(n, n-1) == math.comb(n, 2)
        assert generator.s_n_k(n, 0) == 0
        assert generator.s_n_k(n, n+1) == 0
    assert generator.s_n_k(0, 0) == 1


@pytest.mark.parametrize('n', [0, 1, 4, 7, 9])
def test_capped_counts_are_partial_stirling_rows(n):
    """The RGS table's total must equal S(n,1) + ... + S(n,K), and B(n) with no cap."""
    generator = SetPartition()
    for cap in range(0, n+2):
        expected = sum(generator.s_n_k(n, k) for k in range(cap+1))
        assert generator.rgs_completions(n, cap) == expected
    assert generator.rgs_completions(n) == BELL[n]


def test_count_uses_the_target():
    """count() reads the restriction off the generator, as CombinatorialSequence intends."""
    assert fitted(8).count() == BELL[8]
    assert fitted(8, blocks=3).count() == 966            # S(8,3)
    assert fitted(6, max_blocks=3).count() == 122        # S(6,1) + S(6,2) + S(6,3)
    assert fitted(6).partition_function(weight=5) == BELL[5]


def test_table_growth_matches_a_fresh_build():
    """Extending a table in place must give what building it in one go would have."""
    grown = SetPartition()
    grown.make_s_n_k_table(4, 2)
    grown.make_s_n_k_table(9, 3)      # more columns
    grown.make_s_n_k_table(9, 7)      # then more rows
    grown.make_s_n_k_table(12, 9)     # then both again

    fresh = SetPartition()
    fresh.make_s_n_k_table(12, 9)

    assert grown.s_n_k_table == fresh.s_n_k_table


# ---------------------------------------------------------------- iteration

@pytest.mark.parametrize('n', [0, 1, 2, 3, 5, 6])
def test_iterator_enumerates_every_partition_once(n):
    """The iterator must yield all B(n) partitions, including the very first one."""
    walked = list(fitted(n))
    assert len(walked) == BELL[n]
    assert len(set(walked)) == len(walked), 'the iterator repeated a partition'
    assert sorted(walked) == sorted(all_set_partitions(n))


@pytest.mark.parametrize('n,cap', [(4, 2), (5, 3), (6, 2)])
def test_iterator_respects_the_block_cap(n, cap):
    walked = list(fitted(n, max_blocks=cap))
    assert sorted(walked) == sorted(all_set_partitions(n, max_blocks=cap))


@pytest.mark.parametrize('n,k', [(4, 2), (5, 3), (6, 2), (6, 6)])
def test_iterator_respects_an_exact_block_count(n, k):
    generator = SetPartition()
    generator.fit(weight=n, blocks=k)
    walked = list(generator)
    assert all(len(partition) == k for partition in walked)
    assert len(walked) == generator.s_n_k(n, k)
    assert len(set(walked)) == len(walked)


# ---------------------------------------------------------------- transforms

@pytest.mark.parametrize('n', [1, 4, 6])
def test_rgs_round_trip(n):
    """blocks -> string -> blocks is the identity on every partition of [n]."""
    generator = SetPartition()
    for partition in all_set_partitions(n):
        rgs = _transforms.blocks_to_rgs(partition)
        assert rgs[0] == 0
        for i in range(1, n):
            assert rgs[i] <= 1 + max(rgs[:i]), f'{rgs} is not a restricted growth string'
        assert _transforms.rgs_to_blocks(rgs) == partition


def test_transforms_canonicalize_their_input():
    """Blocks handed over in any order come back in canonical form."""
    generator = SetPartition()
    scrambled = [[5, 2], [4], [3, 1]]
    assert generator.canonicalize(scrambled) == ((1, 3), (2, 5), (4,))
    assert generator.to_rgs(scrambled) == (0, 1, 0, 2, 1)
    assert generator.block_sizes(((1, 3), (2, 5), (4,))) == {2: 2, 1: 1}


# ---------------------------------------------------------------- uniformity

@pytest.mark.parametrize('method', METHODS)
def test_sampler_is_uniform(method):
    """A chi-square goodness-of-fit against the uniform distribution on B(n).

    Being a valid partition does not make a sampler correct -- it has to hit every
    partition equally often. n=5 has 52 partitions, small enough to enumerate and to keep
    the expected cell counts comfortably above 5.
    """
    from scipy.stats import chisquare

    n = 5
    support = all_set_partitions(n)
    counts = collections.Counter(draw(fitted(n), method, len(support)*250))

    assert set(counts) <= set(support), (
        f'{method} produced something that is not a partition of [{n}]: '
        f'{set(counts) - set(support)}')

    _, p_value = chisquare([counts.get(partition, 0) for partition in support])
    assert p_value > 0.001, (
        f'{method} does not look uniform over the {len(support)} partitions of [{n}] '
        f'(chi-square p={p_value:.2e})')


@pytest.mark.parametrize('method', EXACT_METHODS)
def test_sampler_is_uniform_on_a_larger_support(method):
    """The rejection-free methods are quick enough to check against 203 partitions of [6].

    They also all pick their move by an exact integer walk -- over the RGS table, the Bell
    array, the Stirling table, or Dobinski's weights cleared of their denominators -- so a
    larger support is where a rounding error in any of that would start to show.
    """
    from scipy.stats import chisquare

    n = 6
    support = all_set_partitions(n)
    counts = collections.Counter(draw(fitted(n), method, len(support)*120))

    assert set(counts) <= set(support), (
        f'{method} produced a non-partition: {set(counts) - set(support)}')
    assert len(counts) == len(support), (
        f'{method} never reached {len(support) - len(counts)} of the {len(support)} '
        f'partitions of [{n}]')

    _, p_value = chisquare([counts.get(partition, 0) for partition in support])
    assert p_value > 0.001, f'{method} not uniform over B({n}) (chi-square p={p_value:.2e})'


@pytest.mark.parametrize('n,cap', [(5, 2), (6, 3)])
def test_capped_table_method_is_uniform(cap, n):
    """Uniform over the partitions it is restricted to, not just inside the cap."""
    from scipy.stats import chisquare

    support = all_set_partitions(n, max_blocks=cap)
    counts = collections.Counter(draw(fitted(n, max_blocks=cap), 'table_only', len(support)*300))

    assert set(counts) <= set(support)
    _, p_value = chisquare([counts.get(partition, 0) for partition in support])
    assert p_value > 0.001, f'capped at {cap} blocks, not uniform (p={p_value:.2e})'


@pytest.mark.parametrize('n,k', [(5, 2), (6, 3)])
def test_stirling_method_is_uniform_given_the_block_count(n, k):
    """Uniform over the partitions into exactly k blocks."""
    from scipy.stats import chisquare

    support = [p for p in all_set_partitions(n) if len(p) == k]
    counts = collections.Counter(draw(fitted(n, blocks=k), 'stirling', len(support)*300))

    assert set(counts) <= set(support)
    assert len(counts) == len(support)
    _, p_value = chisquare([counts.get(partition, 0) for partition in support])
    assert p_value > 0.001, f'exactly {k} blocks, not uniform (p={p_value:.2e})'


def test_block_counts_follow_the_stirling_numbers():
    """How many blocks a sample has must be distributed as S(n,k)/B(n).

    A sampler can be wrong in a way that keeps every sample valid but skews the number of
    blocks, and at n=20 the support is far too big to test partition by partition.
    """
    from scipy.stats import chisquare

    n = 20
    generator = fitted(n)
    draws = 20000
    expected = [draws*generator.s_n_k(n, k)/BELL_20 for k in range(1, n+1)]

    for method in EXACT_METHODS:
        counts = collections.Counter(len(p) for p in draw(generator, method, draws))
        observed = [counts.get(k, 0) for k in range(1, n+1)]

        # Pool the tails, where the expected counts are too small for chi-square.
        pooled_observed, pooled_expected = [], []
        for count, mean in zip(observed, expected):
            if pooled_expected and pooled_expected[-1] < 5:
                pooled_observed[-1] += count
                pooled_expected[-1] += mean
            else:
                pooled_observed.append(count)
                pooled_expected.append(mean)
        if len(pooled_expected) > 1 and pooled_expected[-1] < 5:
            pooled_observed[-2] += pooled_observed.pop()
            pooled_expected[-2] += pooled_expected.pop()

        # chisquare wants the two to total the same; rounding in the expected counts
        # otherwise leaves them a hair apart.
        scale = sum(pooled_observed)/sum(pooled_expected)
        pooled_expected = [mean*scale for mean in pooled_expected]

        _, p_value = chisquare(pooled_observed, pooled_expected)
        assert p_value > 0.001, (
            f'{method} block counts do not follow S({n},k)/B({n}) (p={p_value:.2e})')


BELL_20 = 51724158235372


# ---------------------------------------------------------------- Stam's truncation

@pytest.mark.parametrize('n', [1, 2, 5, 12, 30])
def test_dobinski_weights_are_the_right_series(n):
    """The integer weights must be k^n/k! cleared of denominators, and reach far enough.

    Stam draws the number of urns from an infinite series, so it has to be cut off; the cut
    is placed where the terms have fallen to 2^-256 of the peak.  Two things to check: the
    weights really are proportional to k^n/k!, and the truncated total is within rounding
    of e*B(n), which is what the whole series sums to by Dobinski's formula.
    """
    weights = _sampling.dobinski_weights(n)
    cutoff = len(weights) - 1

    scale = math.factorial(cutoff)
    for k in range(cutoff+1):
        assert weights[k]*math.factorial(k) == k**n * scale, f'weight {k} is not k^n/k!'

    assert cutoff >= 2*n + 4, 'truncated before the terms are guaranteed to be decaying'

    generator = SetPartition()
    assert sum(weights)/scale == pytest.approx(math.e*generator.bell(n), rel=1e-12)
