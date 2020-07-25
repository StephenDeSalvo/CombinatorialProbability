
import numpy

from CombinatorialProbability.combinatorics import CombinatorialStructure


def fit(self, **kwargs):

    self.target = {}
    if 'weight' in kwargs:
        n = kwargs['weight']
        self.target['n'] = n
        self.n_ = n
        CombinatorialStructure.initialize(self, {n:1})

    if 'components' in kwargs:
        k = kwargs['components']
        self.target['k'] = k
        self.k_ = k
        block_size = int(n/k)
        residual = n - block_size*k
        partition_multiplicities = {k:block_size}
        if residual > 0:
            partition_multiplicities[residual] = 1

        CombinatorialStructure.initialize(self, partition_multiplicities)

    if 'max_part_size' in kwargs:
        ell = kwargs['max_part_size']
        self.target['ell'] = ell
        self.ell_ = ell
        block_size = int(n/k)
        residual = n - block_size*k
        partition_multiplicities = {k:block_size}
        if residual > 0:
            partition_multiplicities[residual] = 1
        CombinatorialStructure.initialize(self, partition_multiplicities)


    make_array = None if 'make_array' not in kwargs else kwargs['make_array']
    array_size = None if 'array_size' not in kwargs else kwargs['array_size']
    make_table = None if 'make_table' not in kwargs else kwargs['make_table']
    table_size = None if 'table_size' not in kwargs else kwargs['table_size']

    if make_array is not None:
        #if len(self.target.keys()) == 1 and 'n' in self.target:
            # standard p(n) formula

        size_of_array = numpy.max([self.target['n'], 0 if array_size is None else array_size])
        self.make_p_of_n_array(size_of_array, **kwargs)


    if make_table is not None:
        #if len(self.target.keys()) == 2 and 'n' in self.target and 'k' in self.target:

        columns = self.target['n']
        rows =  table_size if table_size is not None else self.target['n'] if 'k' not in self.target else self.target['k']

        self.make_p_n_k_table(columns, rows, **kwargs)


    if 'make_tilt' in kwargs:
        tilt = kwargs['make_tilt']
    else:
        tilt = None

    if tilt is not None:
        if len(self.target.keys()) == 1 and 'n' in self.target:
            self.x_ = numpy.exp(-numpy.pi / numpy.sqrt(6*self.target['n']))
            #self.target['tilt'] = self.x_


    return self
