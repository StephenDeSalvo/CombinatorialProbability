""" @file _iterator_methods.py
    @brief Implementation of methods related to combinatorial structures and iteration.

    Set partitions are enumerated in lexicographic order on their restricted growth strings,
    the same order the table method unranks in.  For [3] that is

        000 -> 001 -> 010 -> 011 -> 012

    i.e., ((1,2,3),), ((1,2),(3,)), ((1,3),(2,)), ((1,),(2,3)), ((1,),(2,),(3,)).
"""


from CombinatorialProbability.set_partition._transforms import blocks_to_rgs, rgs_to_blocks


def next_object(self, blocks, **kwargs):
    """Given a set partition, generates the next object in the sequence.

    Returns (next partition, exhausted flag), which is what CombinatorialStructure expects.

    A value of None for the current object means iteration has not started yet, and the
    first partition -- everything in one block -- is returned.  The generator is fit with
    None for exactly this reason: were it primed with the first partition instead, the
    iterator would advance past it before yielding anything and that partition would never
    be seen.

    Restrictions recorded on the target are respected: max_blocks caps how many blocks a
    partition may have, and blocks=k asks for exactly k, which is handled by skipping the
    partitions in between.
    """

    n = self.target['n']
    cap = self.target.get('max_blocks')
    cap = n if cap is None else min(int(cap), n)
    exactly = self.target.get('k')

    if blocks is None:
        # The all-zero string, i.e., a single block, is first in lexicographic order.
        rgs = [0]*n
        exhausted = False
    else:
        rgs = list(blocks_to_rgs(blocks))
        rgs, exhausted = next_rgs(rgs, cap)

    # For an exact block count, walk on until one turns up.  Cheap enough for the sizes
    # this iterator is usable at, and it keeps the ordering the same as everywhere else.
    if exactly is not None:
        while not exhausted and (0 if n == 0 else max(rgs) + 1) != int(exactly):
            rgs, exhausted = next_rgs(rgs, cap)

    return rgs_to_blocks(rgs), exhausted


def next_rgs(rgs, cap):
    """The next restricted growth string in lexicographic order, and whether we ran out.

    Position i may hold any digit from 0 up to min(1 + max(rgs[:i]), cap - 1): a digit
    below that opens no new block, and the one at 1 + max(rgs[:i]) opens the next one,
    which the cap may forbid.  So find the last position that can be raised, raise it, and
    zero everything after it -- zeros are always legal.

    The limits are read off the current string, and only from the prefix before each
    position, so raising position i leaves them all valid.
    """

    n = len(rgs)

    limits = []
    blocks_open = 0
    for i in range(n):
        limits.append(min(blocks_open, cap-1))
        blocks_open = max(blocks_open, rgs[i]+1)

    for i in range(n-1, -1, -1):
        if rgs[i] < limits[i]:
            rgs[i] += 1
            for j in range(i+1, n):
                rgs[j] = 0
            return rgs, False

    return rgs, True
