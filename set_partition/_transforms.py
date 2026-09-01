""" @file _transforms.py
    @brief Conversions between the two ways of writing a set partition.

    A set partition of [n] = {1, 2, ..., n} is carried around here in canonical block form:
    a tuple of tuples, blocks ordered by their least element, elements ascending within a
    block.  So the five set partitions of [3] are

        ((1, 2, 3),)      ((1, 2), (3,))      ((1, 3), (2,))      ((1,), (2, 3))      ((1,), (2,), (3,))

    The other representation is the restricted growth string (RGS): rgs[i-1] is the index
    of the block holding element i, with the rule rgs[0] = 0 and rgs[i] <= 1 + max(rgs[:i]).
    Blocks are then numbered in order of first appearance, which is exactly the canonical
    order above, so the two forms are in bijection.  The RGS is what the unranking sampler
    and the iterator work in, because lexicographic order on strings is an order on
    partitions that a table can count.

    The functions here are plain module-level functions so the samplers and the tests can
    use them without an instance; the class imports thin method wrappers.
"""


def canonical_blocks(blocks):
    """Blocks ordered by least element, elements ascending within each block.

    Empty blocks are dropped first, so that a method which works with a fixed number of
    boxes and only afterwards discovers which are occupied -- Stam's urn algorithm -- can
    hand its boxes straight over.
    """
    occupied = [block for block in blocks if block]
    return tuple(tuple(sorted(block)) for block in sorted(occupied, key=min))


def rgs_to_blocks(rgs):
    """Restricted growth string -> canonical block form.

    Element i (1-based) goes into block rgs[i-1].  Because a valid RGS only ever opens
    block m after every block below m has appeared, the blocks come out already ordered
    by least element, and elements come out ascending.
    """
    blocks = []
    for element, block_index in enumerate(rgs, start=1):
        while block_index >= len(blocks):
            blocks.append([])
        blocks[block_index].append(element)
    return tuple(tuple(block) for block in blocks)


def blocks_to_rgs(blocks):
    """Canonical block form -> restricted growth string.

    The input is canonicalized first, so this is a true inverse of rgs_to_blocks even if
    the caller hands over blocks in some other order.
    """
    blocks = canonical_blocks(blocks)
    n = sum(len(block) for block in blocks)
    rgs = [0]*n
    for block_index, block in enumerate(blocks):
        for element in block:
            rgs[element-1] = block_index
    return tuple(rgs)


def block_size_partition(blocks):
    """The integer partition of n underneath a set partition of [n].

    Returned as {block size: multiplicity}, which is the same shape IntegerPartition
    uses, so a set partition sample can be fed straight into those tools.
    """
    sizes = {}
    for block in blocks:
        sizes[len(block)] = sizes.get(len(block), 0) + 1
    return sizes


# ------------------------------------------------------- method wrappers

def to_blocks(self, rgs):
    """Method form of rgs_to_blocks."""
    return rgs_to_blocks(rgs)


def to_rgs(self, blocks):
    """Method form of blocks_to_rgs."""
    return blocks_to_rgs(blocks)


def canonicalize(self, blocks):
    """Method form of canonical_blocks."""
    return canonical_blocks(blocks)


def block_sizes(self, blocks):
    """Method form of block_size_partition."""
    return block_size_partition(blocks)
