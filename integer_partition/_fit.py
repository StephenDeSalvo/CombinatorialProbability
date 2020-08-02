""" @file _fit.py
    @brief Modules related to precomputing, i.e., "fit"-ting relevant parameters for integer partitions.

"""

import numpy
from CombinatorialProbability.combinatorics import CombinatorialStructure


def fit(self, **kwargs):
    """Precomputes various quantities associated with integer partitions of a given target.

    Right now weight is the only one which works well and has been tested.

    Arguments:
        kwargs: (dict) contains various options for restricting integer partitions by target.

    """

    # A potentially multi-dimensional set of targets for the integer partition to satisfy.
    self.target = {}
    
    # The weight is the total sum of all elements within the integer partition
    if 'weight' in kwargs:
        n = kwargs['weight']
        self.target['n'] = n
        self.n_ = n
        CombinatorialStructure.initialize(self, {n:1})

    # The components is the total number of elements within the integer partition
    # if 'components' in kwargs:
    #     ell = kwargs['components']
    #     self.target['ell'] = ell
    #     self.ell_ = ell
    #     block_size = int(n/ell)
    #     residual = n - block_size*ell
    #     partition_multiplicities = {ell:block_size}
    #     if residual > 0:
    #         partition_multiplicities[residual] = 1

    #     CombinatorialStructure.initialize(self, partition_multiplicities)

    # The max part size is related to the number of components by conjugation.
    # (both max part size and components can be specified.)
    if 'max_part_size' in kwargs:
        k = kwargs['max_part_size']
        self.target['k'] = k
        self.k_ = k
        block_size = int(n/k)
        residual = n - block_size*k
        partition_multiplicities = {k:block_size}
        if residual > 0:
            partition_multiplicities[residual] = 1
        CombinatorialStructure.initialize(self, partition_multiplicities)

    # Flags which dictate whether to precompute the tables or tilting parameters
    make_array = None if 'make_array' not in kwargs else kwargs['make_array']
    array_size = None if 'array_size' not in kwargs else kwargs['array_size']
    make_table = None if 'make_table' not in kwargs else kwargs['make_table']
    table_rows = None if 'table_rows' not in kwargs else kwargs['table_rows']
    tilt = None if 'make_tilt' not in kwargs else kwargs['make_tilt']


    if make_array is not None:

        # if no size is specified, use weight
        size_of_array = numpy.max([self.target['n'], 0 if array_size is None else array_size])
        self.make_p_of_n_array(size_of_array, **kwargs)


    if make_table is not None:

        columns = self.target['n']

        # if no size is specified, use weight and max part size (by default weight)
        table_rows =  int(table_rows) if table_rows is not None else self.target['n'] if 'k' not in self.target else self.target['k']

        self.make_p_n_k_table(columns, table_rows, **kwargs)


    if tilt is not None:

        # 1D tilting parameter
        if len(self.target.keys()) == 1 and 'n' in self.target:
            self.x_ = numpy.exp(-numpy.pi / numpy.sqrt(6*self.target['n']))

        # TODO: 2D tilting parameter, etc.  (See Roth-Szekeres paper, e.g., on the implicitly defined solution)
    


    return self
