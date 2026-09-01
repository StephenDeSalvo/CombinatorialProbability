""" @file _sampling.py
    @brief Functions related to random sampling of set partitions.

    A collection of functions which each generate a uniformly random set partition of
    [n] = {1, ..., n} by a different method.  The sampling function is the one to call,
    with method='rejection' specified, for example.

    Every method returns [sample_list, count_list], the same contract the integer partition
    package uses: the samples, and the number of iterations (1 + number of rejections) each
    one took, which is all 1s for the methods that never reject.

    Samples come back in canonical block form -- ((1, 3), (2,)) and so on -- see _transforms.
"""


# Arbitrary precision random integer.  DO NOT USE scipy or numpy's version as they are only
# int64!  B(n) passes 2^63 by n = 26, and every unranking method here draws a rank uniformly
# from 1..B(n), so an int64 draw would silently stop being uniform almost immediately.
from random import randint, randrange, sample, shuffle

# Floats only for locating the truncation point in Stam; nothing here draws through one.
from math import lgamma, log

import numpy

from scipy.special import lambertw

from CombinatorialProbability.set_partition._transforms import canonical_blocks, rgs_to_blocks


def tilting_parameter(n):
    """The x that makes a Boltzmann sample have expected size exactly n.

    The EGF for set partitions is exp(e^x - 1), so a Boltzmann sample of parameter x has
    expected size x*C'(x)/C(x) = x*e^x.  Setting that to n gives x = W(n), the Lambert W
    function -- the saddle point, and the choice that makes rejection as cheap as it can be.
    """

    return 0.0 if n <= 0 else float(lambertw(n).real)


def sampling(self, **kwargs):
    """Sets up the parameters for a given method and invokes it.

    The methods currently implemented are as follows:
        1.  Rejection/Boltzmann sampling.
        2.  Stam's urn algorithm, via Dobinski's formula.
        3.  Table method - unranking a restricted growth string in lexicographic order.
        4.  Array method - Nijenhuis and Wilf's RANEQU, i.e., the Bell recursion.
        5.  Stirling method - by number of blocks, and the only one that can be asked for
            a partition into exactly k blocks.

    Methods can also have optionally specified parameters, e.g., method_params={'max_blocks':3}
    for the table method.
    """

    method = 'rejection' if 'method' not in kwargs else kwargs['method']
    method_params = kwargs.get('method_params', {})

    n = self.target['n']
    kwargs['target'] = n

    # 'pdc-recursive' and 'pdc_recursive' should reach the same branch.
    normalized = method.lower().replace('-', ' ').replace('_', ' ')

    # A cap on the number of blocks can come from the method parameters or, if the generator
    # was fit with one, from the target.  'rows' is accepted as an alias because it is the
    # row index of the table here in the same way it is for integer partitions.
    cap = method_params.get('max_blocks', method_params.get('rows'))
    if cap is None:
        cap = self.target.get('max_blocks')

    exact_blocks = method_params.get('blocks', self.target.get('k'))

    # A restriction only one method knows how to honour must not be quietly dropped by the
    # others: that would sample uniformly from the wrong set and look entirely healthy.
    capped = cap is not None and int(cap) < n

    def refuse(restriction):
        raise ValueError(
            f"method '{method}' cannot sample set partitions restricted by {restriction} -- "
            "use method='table_only' for max_blocks, or method='stirling' for an exact "
            "number of blocks")

    # Standard rejection sampling: generate a Boltzmann object until it has size exactly n.
    if normalized in ['rejection', 'boltzmann']:

        if capped:
            refuse('max_blocks')
        if exact_blocks is not None:
            refuse('blocks')

        if not hasattr(self, 'x_'):
            self.x_ = tilting_parameter(n)
        kwargs['tilt'] = self.x_

        return rejection_sampling(**kwargs)

    # Stam's urn algorithm: draw the number of urns from Dobinski's formula, then throw the
    # n elements into them independently and uniformly.
    elif normalized in ['stam', 'dobinski', 'urn']:

        if capped:
            refuse('max_blocks')
        if exact_blocks is not None:
            refuse('blocks')

        return stam_sampling(**kwargs)

    # Table method: unrank a restricted growth string in lexicographic order.
    elif normalized in ['recursive', 'nijenhuis wilf', 'table method', 'table only',
                        'unrank', 'rgs']:

        if exact_blocks is not None:
            refuse('blocks')

        kwargs['max_blocks'] = cap
        kwargs['table'] = self.make_rgs_table(n, cap)

        return self.table_method_sampling(**kwargs)

    # Array method, using the Bell recursion on the block containing the least element.
    elif normalized in ['array only', 'bell', 'raneq u', 'raneq']:

        if capped:
            refuse('max_blocks')
        if exact_blocks is not None:
            refuse('blocks')

        self.make_bell_array(n)
        kwargs['array'] = self.bell

        return array_method_sampling(**kwargs)

    # Stirling method: pick the number of blocks, then a partition into exactly that many.
    elif normalized in ['stirling', 'blocks', 'exact blocks', 'block count']:

        if capped:
            refuse('max_blocks')

        blocks = exact_blocks
        kwargs['blocks'] = blocks

        self.make_s_n_k_table(n, n if blocks is None else int(blocks))
        kwargs['table'] = self.s_n_k_table

        return self.stirling_method_sampling(**kwargs)

    raise ValueError(f'unknown method for set partitions: {method}')


# As with integer partitions, the sampling methods are not class methods except where the
# method needs a table, which the generator owns and allocates on demand.


def rejection_sampling(**kwargs):
    """Generates a Boltzmann object of class SET(SET_{>=1}(Z)) until its size is n.

    The labeled Boltzmann sampler for set partitions: the number of blocks M is Poisson
    with mean e^x - 1, each block's size is Poisson(x) conditioned to be at least 1, and
    the n labels are then dealt out by a uniform random permutation.  Conditioned on the
    total size being n, the result is uniform over the set partitions of [n]; the Poisson
    number of blocks is exactly what accounts for the blocks being an unordered set.

    Any x in (0, infinity) is valid, but x = W(n) centres the size on n and so minimises
    the rejection rate.

    kwargs needs 'target' and 'tilt', and optionally 'size' for the number of samples.
    """

    size = 1 if 'size' not in kwargs else kwargs['size']

    n = int(kwargs['target'])
    x = kwargs['tilt']

    sample_list = []
    count_list = []

    for _ in range(size):

        if n == 0:
            sample_list.append(())
            count_list.append(1)
            continue

        counts = 0
        block_sizes = None
        while block_sizes is None:
            counts += 1
            block_sizes = boltzmann_block_sizes(x, n)

        # Deal the labels out.  Which labels land in which block is a uniform permutation,
        # cut into consecutive runs of the sizes just drawn.
        labels = list(range(1, n+1))
        shuffle(labels)

        blocks = []
        position = 0
        for block_size in block_sizes:
            blocks.append(labels[position:position+block_size])
            position += block_size

        sample_list.append(canonical_blocks(blocks))
        count_list.append(counts)

    return [sample_list, count_list]


def boltzmann_block_sizes(x, n):
    """One Boltzmann draw of the block sizes, or None if it does not total n.

    Bails out as soon as the running total passes n rather than finishing the draw, since
    the object is already too big to be accepted.
    """

    number_of_blocks = numpy.random.poisson(numpy.exp(x) - 1)

    block_sizes = []
    total = 0
    for _ in range(number_of_blocks):
        # Poisson(x) conditioned to be at least 1: a block of a set partition is nonempty.
        block_size = 0
        while block_size == 0:
            block_size = int(numpy.random.poisson(x))
        block_sizes.append(block_size)
        total += block_size
        if total > n:
            return None

    return block_sizes if total == n else None


def stam_sampling(**kwargs):
    """Generates samples by Stam's urn algorithm (Stam, JCTA 1983).

    Draw the number of urns N with P(N = k) proportional to k^n / k!, then throw each of
    the n elements into an urn independently and uniformly, and read off the nonempty urns.

    That this is uniform is Dobinski's formula in disguise: a partition with j blocks arises
    from urn count k in (k)_j ways out of k^n, so its probability is proportional to
    sum_k (k)_j/k! = sum_{k>=j} 1/(k-j)! = e, the same for every j.  The block count never
    enters, so every partition is equally likely.  There is no rejection.

    kwargs needs 'target', and optionally 'size' for the number of samples.
    """

    size = 1 if 'size' not in kwargs else kwargs['size']

    n = int(kwargs['target'])

    weights = dobinski_weights(n)

    # Cumulative once, not once per sample.  These are exact integers, so the draw is a
    # walk over whole numbers rather than a comparison against normalized floats.
    cumulative_weights = []
    running = 0
    for weight in weights:
        running += weight
        cumulative_weights.append(running)
    total = cumulative_weights[-1]

    sample_list = []

    for _ in range(size):

        variate = randint(1, total)
        urns = 0
        while cumulative_weights[urns] < variate:
            urns += 1

        boxes = [[] for _ in range(urns)]
        for element in range(1, n+1):
            boxes[randrange(urns)].append(element)

        sample_list.append(canonical_blocks(boxes))

    return [sample_list, [1]*size]


def dobinski_weights(n):
    """Integer weights proportional to k^n/k! for k = 0, 1, ..., K.

    The series sum_k k^n/k! = e*B(n) is infinite, so it has to be cut off somewhere.  Past
    k = 2n + 4 the ratio of consecutive terms is (1 + 1/k)^n/(k+1) <= e^(n/k)/(k+1) < 1/2,
    so terms at least halve and the whole tail beyond K is under twice the term at K+1.  K
    is pushed out until that term is 2^-256 of the largest one, which puts the truncation
    far below the resolution of anything downstream.

    Multiplying every term by K! clears the denominators: k^n * (K!/k!) is an integer, and
    the resulting distribution is exact, no floating point anywhere in the draw itself.
    """

    if n == 0:
        # k^0/k! = 1/k!, and every urn count yields the same object, the empty partition.
        return [1]

    def log_weight(k):
        return n*log(k) - lgamma(k+1)

    cutoff = max(2*n + 4, 8)
    peak = max(log_weight(k) for k in range(1, cutoff+1))
    while log_weight(cutoff) > peak - 256*log(2):
        cutoff += 1

    # ratios[k] = cutoff!/k!, built downwards so each is one multiplication.
    ratios = [0]*(cutoff+1)
    running = 1
    for k in range(cutoff, -1, -1):
        ratios[k] = running
        running *= k

    return [k**n * ratios[k] for k in range(cutoff+1)]


def array_method_sampling(**kwargs):
    """Generates samples according to Nijenhuis and Wilf's Combinatorial Algorithms,
    Algorithm RANEQU (Page 96).

    Utilizes the Bell recursion B(m) = sum_k C(m-1, k-1) B(m-k) to build the partition one
    block at a time: the block containing the least remaining element has size k with
    weight C(m-1, k-1) B(m-k), and its other k-1 members are a uniform subset of what is
    left.  Peeling that block off leaves a uniform partition of the rest.

    As with the integer partition array method, picking the block size is a walk along
    cumulative weights in whole numbers.  B(m) is an arbitrary-precision integer and turning
    these weights into probabilities would put a huge integer over a huge integer and round
    the answer -- B(m) exceeds 2^53 by m = 22, well inside the range this is used for.

    kwargs needs 'target' and 'array' (a callable giving B(m)), optionally 'size'.
    """

    size = 1 if 'size' not in kwargs else kwargs['size']
    array = kwargs['array']

    n = int(kwargs['target'])

    # B(0..n) once up front, not per block chosen per sample.
    bell_numbers = [array(m) for m in range(n+1)]

    sample_list = []

    for _ in range(size):

        remaining = list(range(1, n+1))
        blocks = []

        while remaining:
            m = len(remaining)
            variate = randint(1, bell_numbers[m])

            # binomial is C(m-1, block_size-1), carried along by the multiplicative
            # recurrence rather than recomputed: comb() in the innermost loop is the same
            # cost the integer partition array method used to pay.
            cumulative = 0
            binomial = 1
            for block_size in range(1, m+1):
                cumulative += binomial*bell_numbers[m-block_size]
                if cumulative >= variate:
                    break
                binomial = binomial*(m - block_size)//block_size

            # The least remaining element is always in this block; the rest of it is a
            # uniform (block_size - 1)-subset of the others.
            least = remaining[0]
            companions = sample(remaining[1:], block_size-1)
            block = sorted([least] + companions)
            blocks.append(block)

            spent = set(block)
            remaining = [element for element in remaining if element not in spent]

        sample_list.append(canonical_blocks(blocks))

    return [sample_list, [1]*size]


def table_method_sampling(self, **kwargs):
    """Generates samples by unranking a restricted growth string in lexicographic order.

    The RGS of a set partition of [n] records, for each element in turn, which block it
    joins; the rule is that element 1 opens block 0 and no element may open block m+1
    before block m exists.  Lexicographic order on those strings is an order on the set
    partitions themselves, and the table counts how many strings lie under each prefix.

    Numbering.  With j positions still to fill and m blocks open, the strings continuing
    with an existing block d occupy the first m*table[j-1][m] ranks, table[j-1][m] apiece
    in the order d = 0, 1, ..., m-1, and the strings that open block m occupy the
    table[j-1][m+1] ranks after them.  Decoding a rank is therefore one division per
    element: the quotient names the block, the remainder is the rank within it.

    If kwargs has a max_blocks parameter, it samples from the set partitions of [n] into at
    most that many blocks; the table built with that cap simply never counts a string that
    opens one block too many, so the cap needs no separate check while decoding.
    """

    size = 1 if 'size' not in kwargs else kwargs['size']

    n = int(kwargs['target'])
    cap = kwargs.get('max_blocks')
    table = kwargs.get('table')
    if table is None:
        table = self.make_rgs_table(n, cap)

    sample_list = []

    for _ in range(size):

        # The valid ranks are 1..table[n][0].  random.randint is inclusive at BOTH ends.
        variate = randint(1, table[n][0]) if n > 0 else 1

        rgs = []
        blocks_open = 0

        for positions_left in range(n, 0, -1):

            if blocks_open == 0:
                # Element 1 has no choice: it opens block 0.  Its weight is the whole of
                # table[n][0], so the rank passes through untouched.
                rgs.append(0)
                blocks_open = 1
                continue

            completions = table[positions_left-1][blocks_open]
            quotient, remainder = divmod(variate - 1, completions)

            if quotient < blocks_open:
                # Joins an existing block, and the leftover is the rank within it.
                rgs.append(quotient)
                variate = remainder + 1
            else:
                # Past all the existing-block ranks, so this element opens a new block.
                variate -= blocks_open*completions
                rgs.append(blocks_open)
                blocks_open += 1

        sample_list.append(rgs_to_blocks(rgs))

    return [sample_list, [1]*size]


def stirling_method_sampling(self, **kwargs):
    """Generates samples by way of the number of blocks, using S(n,k).

    Two steps.  The number of blocks K is drawn with P(K = k) = S(n,k)/B(n), unless the
    caller has fixed it, and then a partition into exactly k blocks is built by running the
    recursion S(i,k) = k*S(i-1,k) + S(i-1,k-1) downwards: element i either joins one of the
    k blocks formed by the elements below it, with weight k*S(i-1,k), or opens the k-th
    block itself, with weight S(i-1,k-1).

    Which of the k blocks a joining element lands in is not known while descending, since
    the blocks do not exist yet -- so the choice is recorded as an index and applied on the
    way back up, where the blocks are created in order of least element and the index means
    what it should.  The single draw at each step supplies both the branch and, by its
    quotient, the uniform choice among the k blocks.

    This is the only method that can be asked for exactly k blocks: kwargs may carry
    'blocks', or the generator may have been fit with blocks=k.
    """

    size = 1 if 'size' not in kwargs else kwargs['size']

    n = int(kwargs['target'])
    requested_blocks = kwargs.get('blocks')
    requested_blocks = None if requested_blocks is None else int(requested_blocks)

    table = kwargs.get('table')
    if table is None:
        self.make_s_n_k_table(n, n if requested_blocks is None else requested_blocks)
        table = self.s_n_k_table

    if requested_blocks is not None:
        if requested_blocks < 0 or requested_blocks > n or (requested_blocks == 0 and n > 0):
            raise ValueError(
                f'there are no set partitions of [{n}] into exactly {requested_blocks} blocks')

    # P(K = k) proportional to S(n,k).  Exact integers, so this is a walk, not a normalization.
    if requested_blocks is None:
        block_count_weights = []
        running = 0
        for k in range(n+1):
            running += table[k][n]
            block_count_weights.append(running)

    sample_list = []

    for _ in range(size):

        if requested_blocks is None:
            variate = randint(1, block_count_weights[-1])
            blocks_wanted = 0
            while block_count_weights[blocks_wanted] < variate:
                blocks_wanted += 1
        else:
            blocks_wanted = requested_blocks

        # Descend, recording for each element either None (it opens a block) or the index
        # of the block it joins among those formed by the elements below it.
        decisions = []
        blocks_left = blocks_wanted
        for element in range(n, 0, -1):
            joining_weight = blocks_left*table[blocks_left][element-1]
            variate = randint(1, table[blocks_left][element])
            if variate <= joining_weight:
                decisions.append(divmod(variate-1, table[blocks_left][element-1])[0])
            else:
                decisions.append(None)
                blocks_left -= 1

        # Ascend.  Blocks are created in order of least element, so after elements
        # 1..i-1 the list holds exactly the blocks the recorded index was counting.
        blocks = []
        for element, decision in zip(range(1, n+1), reversed(decisions)):
            if decision is None:
                blocks.append([element])
            else:
                blocks[decision].append(element)

        sample_list.append(canonical_blocks(blocks))

    return [sample_list, [1]*size]
