""" @file _fit.py
    @brief Modules related to precomputing, i.e., "fit"-ting relevant parameters for set partitions.

"""

from CombinatorialProbability.combinatorics import CombinatorialStructure
from CombinatorialProbability.set_partition._sampling import tilting_parameter


def fit(self, **kwargs):
    """Precomputes various quantities associated with set partitions of a given target.

    Arguments:
        kwargs: (dict) contains various options for restricting set partitions by target.

    The size of the ground set is 'weight', named to match the integer partition package,
    since a set partition of [n] is a labeled structure of size n.  'n' and 'elements' are
    accepted as aliases -- note that 'size' is not, because sampling() already uses that
    for the number of samples to draw.
    """

    self.target = {}

    n = None
    for key in ('weight', 'n', 'elements'):
        if key in kwargs:
            n = int(kwargs[key])
            break

    if n is None:
        raise ValueError("fit needs the size of the ground set, e.g., fit(weight=10)")
    if n < 0:
        raise ValueError(f'the ground set cannot have {n} elements')

    self.target['n'] = n
    self.n_ = n

    # Exactly k blocks.  Only the Stirling method can sample under this restriction; it is
    # also honoured by counting and by iteration.
    if 'blocks' in kwargs:
        k = int(kwargs['blocks'])
        self.target['k'] = k
        self.k_ = k

    # At most K blocks.  This is the restriction the table method takes, in the same way
    # the integer partition table method takes a cap on part size.
    if 'max_blocks' in kwargs:
        self.target['max_blocks'] = int(kwargs['max_blocks'])

    # None means "iteration has not started", so that the first partition is yielded rather
    # than stepped over.  See _iterator_methods.next_object.
    CombinatorialStructure.initialize(self, None)

    # Flags which dictate whether to precompute the tables or tilting parameters
    make_array = None if 'make_array' not in kwargs else kwargs['make_array']
    array_size = None if 'array_size' not in kwargs else kwargs['array_size']
    make_table = None if 'make_table' not in kwargs else kwargs['make_table']
    table_rows = None if 'table_rows' not in kwargs else kwargs['table_rows']
    tilt = None if 'make_tilt' not in kwargs else kwargs['make_tilt']

    # Nothing below forwards kwargs: the builders take n, k and max_blocks positionally and
    # the caller's own kwargs use those same names, which would collide.
    if make_array is not None:

        # if no size is specified, use the size of the ground set
        size_of_array = max(n, 0 if array_size is None else int(array_size))
        self.make_bell_array(size_of_array)

    if make_table is not None:

        # if no size is specified, use the ground set and the block restriction, which by
        # default allows every block count.
        rows = int(table_rows) if table_rows is not None else self.target.get('k', n)

        self.make_s_n_k_table(n, rows)
        self.make_rgs_table(n, self.target.get('max_blocks'))

    if tilt is not None:

        # The saddle point x*e^x = n for the Boltzmann sampler.
        self.x_ = tilting_parameter(n)

    return self
