

# import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '../..', ''))

#import numpy



from CombinatorialProbability.combinatorics import CombinatorialSequence
from CombinatorialProbability.combinatorics import CombinatorialStructure



class IntegerPartition(CombinatorialSequence, CombinatorialStructure):
    def __init__(self, **kwargs):
        #super(integer_partition, self).__init__(self)
        CombinatorialSequence.__init__(self, self)
        CombinatorialStructure.__init__(self, self)

        self.p_of_n_array = None
        self.p_n_k_table = None
        self.target = {}
        self.part_sizes = None

        if 'part_sizes' in kwargs:
            self.part_sizes = kwargs['part_sizes']



    from CombinatorialProbability.integer_partition._make_table import make_p_n_k_table
    from CombinatorialProbability.integer_partition._make_table import p_n_k

    from CombinatorialProbability.integer_partition._make_table import make_p_of_n_array
    from CombinatorialProbability.integer_partition._make_table import p_of_n

    from ._fit import fit

    from ._partition_function import partition_function

    from ._sampling import sampling

    from ._iterator_methods import next_object

    from ._transforms import partition_map_to_tuple

    # def fit(self, **kwargs):

    #     self.target = {}
    #     if 'weight' in kwargs:
    #         n = kwargs['weight']
    #         self.target['n'] = n
    #         self.n_ = n
    #         CombinatorialStructure.initialize(self, {n:1})

    #     if 'components' in kwargs:
    #         k = kwargs['components']
    #         self.target['k'] = k
    #         self.k_ = k
    #         block_size = int(n/k)
    #         residual = n - block_size*k
    #         partition_multiplicities = {k:block_size}
    #         if residual > 0:
    #             partition_multiplicities[residual] = 1

    #         CombinatorialStructure.initialize(self, partition_multiplicities)

    #     if 'max_part_size' in kwargs:
    #         ell = kwargs['max_part_size']
    #         self.target['ell'] = ell
    #         self.ell_ = ell
    #         block_size = int(n/k)
    #         residual = n - block_size*k
    #         partition_multiplicities = {k:block_size}
    #         if residual > 0:
    #             partition_multiplicities[residual] = 1
    #         CombinatorialStructure.initialize(self, partition_multiplicities)


    #     make_table = None if 'make_table' not in kwargs else kwargs['make_table']
    #     table_size = None if 'table_size' not in kwargs else kwargs['table_size']

    #     if make_table is not None:
    #         if len(self.target.keys()) == 1 and 'n' in self.target:
    #             # standard p(n) formula

    #             size_of_table = max([self.target['n'], 0 if table_size is None else table_size])

    #             self.__make_p_of_n_table(size_of_table, **kwargs)


    #         if len(self.target.keys()) == 2 and 'n' in self.target and 'k' in self.target:

    #             size_of_table = max([self.target['k'], 0 if table_size is None else table_size])
    #             self.__make_p_n_k_table(self.target['n'], size_of_table, **kwargs)


    #     if 'tilt' in kwargs:
    #         tilt = kwargs['tilt']
    #     else:
    #         tilt = None

    #     if tilt is not None:
    #         if len(self.target.keys()) == 1 and 'n' in self.target:
    #             self.x_ = numpy.exp(-numpy.pi / numpy.sqrt(6*self.target['n']))


    #     return self



    # def partition_function(self, **kwargs):

    #     # Overrides internal value of n stored in target.
    #     if 'weight' in kwargs:
    #         weight = int(kwargs['weight'])
    #         return self.__p_of_n(weight, **kwargs)

    #     if len(self.target.keys()) == 1 and 'n' in self.target:
    #         return self.__p_of_n(self.target['n'], **kwargs)


    #     if len(self.target.keys()) == 2 and 'n' in self.target and 'k' in self.target:
    #         return self.__p_n_k(self.target['n'], self.target['k'], **kwargs)



    # def __make_p_n_k_table(self, n, k, **kwargs):


    #     if self.p_n_k_table is None:
    #         table_columns = numpy.max([n+1, 2])
    #         table_rows = numpy.max([k+1, 2])
    #         self.p_n_k_table = [[None for _ in range(table_columns)] for _ in range(table_rows)]

    #         self.p_n_k_table[0][0] = 1
    #         #print(self.p_n_k_table)
    #         for i in range(1, n+1):
    #             self.p_n_k_table[0][i] = 0

    #         for i in range(1, k+1):
    #             self.p_n_k_table[i][0] = 1

    #         #print(self.p_n_k_table)
    #     else:
    #         if len(self.p_n_k_table[0]) < (n+1):
    #             for i in range(k+1):
    #                 self.p_n_k_table[k] += [None]*(n+1-len(self.p_n_k_table[k]))

    #         if len(self.p_n_k_table) < (k+1):
    #                 self.p_n_k_table += [None]*(n+1)

    #     if self.p_n_k_table[k][n] is None:
    #         for i in range(1, k+1):
    #             for j in range(1, n+1):
    #                 # print('i=', i, ' j=', j)
    #                 # print(self.p_n_k_table)
    #                 if self.p_n_k_table[i][j] is None:
    #                     if i > j:
    #                         self.p_n_k_table[i][j] = self.p_n_k_table[j][j]
    #                     else:
    #                         # print('i=',i, 'j-i=', j-i, 'p(i,j-i) = ', self.p_n_k_table[i][j-i])
    #                         # print('i-1=', i-1, 'j-1=', j-1, 'p(i-1,j-1)=', self.p_n_k_table[i-1][j])
    #                         first_term = 0 if j-i < 0 else self.p_n_k_table[i][j-i]
    #                         second_term = 0 if (i-1 < 0 or j-1 < 0) else self.p_n_k_table[i-1][j]
    #                         # print(first_term, second_term)
    #                         #self.p_n_k_table[i][j] = self.p_n_k_table[i][j-i] + self.p_n_k_table[i-1][j-1]
    #                         self.p_n_k_table[i][j] = first_term + second_term

    #     #return self.p_n_k_table[k][n]

    # def __p_n_k(self, n, k, **kwargs):

    #     if not (self.p_n_k_table is not None and len(self.p_n_k_table) >= k+1 and len(self.p_n_k_table[0]) >= n+1):
    #         self.__make_p_n_k_table(n, k, **kwargs)

    #     return self.p_n_k_table[k][n]


    # def __make_p_of_n_table(self, n, **kwargs):

    #     if 'sequence' in kwargs:
    #         method = kwargs['sequence']
    #     else:
    #         if n <= 10000:
    #             method = 'euler'
    #         else:
    #             method = 'hardy-ramanujan'

    #     if method.lower() == 'euler': #'euler' in kwargs and kwargs['euler'] is True:

    #         if self.p_of_n_table is None:
    #             table_size = numpy.max([n+1, 2])
    #             self.p_of_n_table = [None]*(table_size)
    #             self.p_of_n_table[0] = 1
    #             self.p_of_n_table[1] = 1
    #         else:
    #             if len(self.p_of_n_table) < (n+1):
    #                 self.p_of_n_table += [None]*(n+1-len(self.p_of_n_table))

    #         if self.p_of_n_table[n] is not None:
    #             return self.p_of_n_table[n]

    #         #print('n = ', n)
    #         pn = 0

    #         euler_formula_lower = int(-(numpy.sqrt(24*n+1)-1)/6)
    #         euler_formula_upper = int((numpy.sqrt(24*n+1)+1)/6)
    #         for i in range(euler_formula_lower, euler_formula_upper+1):

    #             index = int(n - i*(3*i-1)/2)
    #             #print(index)
    #             if index < n:
    #                 if self.p_of_n_table[index] is None:
    #                     self.p_of_n_table[index] = self.__make_p_of_n_table(index)

    #                 if i % 2 == 0:
    #                     pn -= self.p_of_n_table[index]
    #                 else:
    #                     pn += self.p_of_n_table[index]

    #         self.p_of_n_table[n] = pn
    #         return pn




    #     elif method.lower() in ['hardy-ramanujan', 'hr', 'hardy ramanujan', 'hardy and ramanujan']:
            
    #         if n <= 3600:
    #             return self.partition_function(weight=n)
    #         else:
    #             return 'n > 10,000 is not currently supported'


    # def __p_of_n(self, n, **kwargs):

    #     if not (self.p_of_n_table is not None and len(self.p_of_n_table) >= n+1):
    #         self.__make_p_of_n_table(n, **kwargs)

    #     return self.p_of_n_table[n]


    # def sampling(self, **kwargs):

    #     size= 1 if 'size' not in kwargs else kwargs['size']

    #     method = 'rejection' if 'method' not in kwargs else kwargs['method']


    #     # Standard rejection sampling: Sample (Z_1, Z_2, ..., Z_n) --> until sum_i i*Z_i = n
    #     if method.lower() in ['rejection', 'boltzmann']:
    #         sample_list = []
    #         count_list = []
    #         n = self.target['n']
    #         x = self.x_
    #         for i in range(size):
    #             partition = {}
    #             counts = 0
    #             while numpy.sum([x*y for x,y in partition.items()]) != n:
    #                 partition = {(i+1):y for i, y in enumerate([geom.rvs(1-x**i)-1 for i in range(1, n+1) if x**i != 1.0]) if y != 0}
    #                 counts += 1

    #             sample_list.append(partition)
    #             count_list.append(counts)

    #         return [sample_list, count_list]

    #     # Probabilistic divide-and-conquer: deterministic second half: Sample (Z_2, Z_3, ..., Z_n) --> Until U < P(Z_1 = n-sum_{i\geq 2} i*z_i) / max_j P(Z_1=j)
    #     elif method.lower() in ['pdcdsh', 'pdc-dsh']:
    #         sample_list = []
    #         count_list = []
    #         n = self.target['n']
    #         x = self.x_
    #         for i in range(size):
    #             partition = {}
    #             counts = 0
    #             keep_going = True
    #             while keep_going is True:
    #                 partition = {(i+2):y for i, y in enumerate([geom.rvs(1-x**i)-1 for i in range(2, n+1) if x**i != 1.0]) if y != 0}
    #                 U = uniform().rvs(size=1)
    #                 residual = n - numpy.sum([x*y for x,y in partition.items()])
    #                 if U < geom.pmf(residual, 1-x, -1) / geom.pmf(0, 1-x, -1):
    #                     keep_going = False
    #                     if residual > 0:
    #                         partition[1] = residual
    #                 counts += 1

    #             sample_list.append(partition)
    #             count_list.append(counts)

    #         return [sample_list, count_list]

    #     elif method.lower() in ['pdcdsh', 'pdc-dsh']:
    #         sample_list = []
    #         count_list = []
    #         n = self.target['n']
    #         x = self.x_
    #         for i in range(size):
    #             partition = {}
    #             counts = 0
    #             keep_going = True
    #             while keep_going is True:
    #                 partition = {(i+2):y for i, y in enumerate([geom.rvs(1-x**i)-1 for i in range(2, n+1) if x**i != 1.0]) if y != 0}
    #                 U = uniform().rvs(size=1)
    #                 residual = n - numpy.sum([x*y for x,y in partition.items()])
    #                 if U < geom.pmf(residual, 1-x, -1) / geom.pmf(0, 1-x, -1):
    #                     keep_going = False
    #                     if residual > 0:
    #                         partition[1] = residual
    #                 counts += 1

    #             sample_list.append(partition)
    #             count_list.append(counts)

    #         return [sample_list, count_list]





    # def next_object(self, part_sizes_list):

    #     flag = False
    #     if numpy.sum(part_sizes_list) == 1:
    #         flag = True

    #     index = len(part_sizes_list)-1
    #     while index >= 0 and part_sizes_list[index] == 1:
    #         index -= 1

    #     if index < 0:
    #         flag = True

    #     part_sizes_list[index] -= 1
    #     part_sizes_list += [1]

    #     if part_sizes_list[index] < 0:
    #         flag = True

    #     return part_sizes_list, flag


    # def next_object(self, component_mulitplicities):
    #     """
    #     Reference
    #     ---------
    #     Modified from Tim Peter's posting to accomodate a k value:
    #     http://code.activestate.com/recipes/218332/
        
    #     Generate all partitions of integer n (>= 0) using integers no
    #     greater than k (default, None, allows the partition to contain n).

    #     Each partition is represented as a multiset, i.e. a dictionary
    #     mapping an integer to the number of copies of that integer in
    #     the partition.  For example, the partitions of 4 are {4: 1},
    #     {3: 1, 1: 1}, {2: 2}, {2: 1, 1: 2}, and {1: 4} corresponding to
    #     [4], [1, 3], [2, 2], [1, 1, 2] and [1, 1, 1, 1], respectively.
    #     In general, sum(k * v for k, v in a_partition.iteritems()) == n, and
    #     len(a_partition) is never larger than about sqrt(2*n).

    #     Note that the _same_ dictionary object is returned each time.
    #     This is for speed:  generating each partition goes quickly,
    #     taking constant time independent of n. If you want to build a list
    #     of returned values then use .copy() to get copies of the returned
    #     values:

    #     >>> p_all = []
    #     >>> for p in partitions(6, 2):
    #     ...         p_all.append(p.copy())
    #     ...
    #     >>> print p_all
    #     [{2: 3}, {1: 2, 2: 2}, {1: 4, 2: 1}, {1: 6}]

    #     """

    #     # if n < 0:
    #     #     raise ValueError("n must be >= 0")

    #     # if n == 0:
    #     #     yield {}
    #     #     return

    #     # if k is None or k > n:
    #     #     k = n

    #     # q, r = divmod(n, k)
    #     # ms = {k : q}
    #     # keys = [k]
    #     # if r:
    #     #     ms[r] = 1
    #     #     keys.append(r)
    #     # yield ms

    #     ms = component_mulitplicities
    #     keys = list(ms.keys())

    #     while keys != [1]:
    #         # Reuse any 1's.
    #         if keys[-1] == 1:
    #             del keys[-1]
    #             reuse = ms.pop(1)
    #         else:
    #             reuse = 0

    #         # Let i be the smallest key larger than 1.  Reuse one
    #         # instance of i.
    #         i = keys[-1]
    #         newcount = ms[i] = ms[i] - 1
    #         reuse += i
    #         if newcount == 0:
    #             del keys[-1], ms[i]

    #         # Break the remainder into pieces of size i-1.
    #         i -= 1
    #         q, r = divmod(reuse, i)
    #         ms[i] = q
    #         keys.append(i)
    #         if r:
    #             ms[r] = 1
    #             keys.append(r)

    #         return ms, False

    #     return ms, True





