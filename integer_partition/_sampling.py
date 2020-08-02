""" @file _sampling.py
    @brief Functions related to random sampling of integer partitions.

    A collection of functions which generate random integer partitions each according to a different
    method.  The sampling function is the one to call with method='rejection' specified, for example.

"""


# Cleaner code to just do my_dict[element] += 1 rather than check if element is in my_dict each time
from collections import defaultdict

# Arbitrary precision random integer.  DO NOT USE scipy or numpy's version as they are only int64!
from random import randint

import numpy

from scipy.stats import geom, uniform

# Used to multiply very large integers with very small floating point numbers
from decimal import Decimal


def sampling(self, **kwargs):
    """Sets up the parameters for a given method and invokes it.

    Each method is set up to return a list of random partitions according to size, and the counts 
    for the number of iterations (i.e., 1+number of rejections) for the given algorithm.  For 
    algorithms which do not have a rejection the counts list is all 1s.

    The methods currently implemented are as follows:
        1.  Rejection/Boltzmann sampling.
        2.  Array method - Nijenhuis and Wilf, i.e., Euler's recursion.
        3.  Table method - Nijenhuis and Wilf, i.e., the recursive method, p(n,k) recursion
        4.  PDCDSH - Probabilistic divide-and-conquer (PDC) deterministic second half, using index i=1
        5.  PDC Recursive - Combination of PDC and and the table method.

    Methods can also have optionally specified parameters, e.g., method_params={'rows':3} for the 
    PDC Recursive method.
    """

    method = 'rejection' if 'method' not in kwargs else kwargs['method']

    # Standard rejection sampling: Sample (Z_1, Z_2, ..., Z_n) --> until sum_i i*Z_i = n
    if method.lower() in ['rejection', 'boltzmann']:

        kwargs['target'] = self.target['n']

        if not hasattr(self, "x_"):
            if len(self.target.keys()) == 1 and 'n' in self.target:
                self.x_ = numpy.exp(-numpy.pi / numpy.sqrt(6*self.target['n']))
        kwargs['tilt'] = self.x_

        #kwargs['distribution'] = geom
        #kwargs['distribution_params'] = {'loc':-1}
        return rejection_sampling(**kwargs)

    # Probabilistic divide-and-conquer: deterministic second half: Sample (Z_2, Z_3, ..., Z_n) --> Until U < P(Z_1 = n-sum_{i\geq 2} i*z_i) / max_j P(Z_1=j)
    elif method.lower() in ['pdcdsh', 'pdc-dsh']:
        kwargs['target'] = self.target['n']

        if not hasattr(self, "x_"):
            if len(self.target.keys()) == 1 and 'n' in self.target:
                self.x_ = numpy.exp(-numpy.pi / numpy.sqrt(6*self.target['n']))
        kwargs['tilt'] = self.x_

        return pdcdsh_sampling(**kwargs)

    # Table method, unrank, or "The recursive method of nijenhuis and Wilf"
    elif method.lower() in ['recursive', 'nijenhuis-wilf', 'table_method', 'table_only', 'unrank']:

        kwargs['target'] = self.target['n']

        n = self.target['n']
        if self.p_n_k_table is None or len(self.p_n_k_table) < n or len(self.p_n_k_table[0]) < n:
            self.make_p_n_k_table(n,n,**kwargs)

        kwargs['table'] = self.p_n_k_table

        # TODO: Implement dynamic allocation of table while respecting table method with inputs of n and k.
        return self.table_method_sampling(**kwargs)

    # Array method, using Euler's recursion.
    elif method.lower() in ['array_only', 'euler', 'divisors']:

        kwargs['target'] = self.target['n']

        n = self.target['n']

        
        if self.p_of_n_array is None or len(self.p_of_n_array) < n:
            self.make_p_of_n_array(n, **kwargs)

        kwargs['array'] = self.p_of_n

        return array_method_sampling(**kwargs)

    # PDC Recursive hybrid method.  Has additional model parameters. 
    elif method.lower().replace('-', ' ').replace('_', ' ') in ['pdc recursive', 'pdc hybrid']:

        kwargs['target'] = self.target['n']

        if not hasattr(self, "x_"):
            if len(self.target.keys()) == 1 and 'n' in self.target:
                self.x_ = numpy.exp(-numpy.pi / numpy.sqrt(6*self.target['n']))
        kwargs['tilt'] = self.x_

        rows = 1 if 'method_params' not in kwargs else 1 if 'rows' not in kwargs['method_params'] else int(kwargs['method_params']['rows'])

        kwargs['rows'] = rows

        n = self.target['n']

        if self.p_n_k_table is None or len(self.p_n_k_table) < rows or len(self.p_n_k_table[0]) < n:
            self.make_p_n_k_table(n,rows,**kwargs)

        kwargs['table'] = self.p_n_k_table

        return self.pdc_recursive_method_sampling(**kwargs)



# Note that these sampling methods are not class methods, except for table method because it dynamically
# allocates a larger table as needed.  This is to avoid having to create the full n x n table and also
# avoid the tediousness of finding the right sized table to create.  

def rejection_sampling(**kwargs):
    """Generates Z_1, Z_2, ..., Z_n until sum_i i*Z_i = kwargs['target']

    Standard rejection sampling from Boltzmann principles and Fristedt.  Z_i is Geometric 1-x^i, where
    x can be any number between 0 and 1, but best to use x=exp(-pi / sqrt(6n)).

    kwargs needs to have 'target', 'tilt', and optionally 'size' for number of samples (by default 1).
    """
    size= 1 if 'size' not in kwargs else kwargs['size']

    n = kwargs['target']
    x = kwargs['tilt']
    
    sample_list = []
    count_list = []

    for i in range(size):
        partition = {}
        counts = 0
        while numpy.sum([x*y for x,y in partition.items()]) != n:

            # Generate vector of uniform random variables
            geom_rvs = [int(numpy.floor(numpy.log(u) / ((i+1)*numpy.log(x)))) for i, u in enumerate(uniform().rvs(n))]
            partition = {(i+1):y for i, y in enumerate(geom_rvs) if y != 0}
            #partition = {(i+1):y for i, y in enumerate([geom.rvs(1-x**i, loc=-1) for i in range(1, n+1) if x**i != 1.0]) if y != 0}
            counts += 1

        sample_list.append(partition)
        count_list.append(counts)

    return [sample_list, count_list]


def pdcdsh_sampling(**kwargs):
    """Generates U_1, Z_2, ..., Z_n until U_1 < P(Z_1 = n - sum_{i>=2} i*Z_i) / P(Z_1 = 0)

    Probabilistic divide-and-conquer deterministic second half method (PDCDSH) by Arratia and DeSalvo.
    Z_i is Geometric 1-x^i, where x can be any number between 0 and 1, but best to use x=exp(-pi / sqrt(6n)).
    For integer partition, using index i=1 is optimal.

    kwargs needs to have 'target', 'tilt', and optionally 'size' for number of samples (by default 1).
    """

    size= 1 if 'size' not in kwargs else kwargs['size']

    sample_list = []
    count_list = []
    n = kwargs['target']
    x = kwargs['tilt']
    for i in range(size):
        partition = {}
        counts = 0
        keep_going = True
        while keep_going is True:

            geom_rvs = [int(numpy.floor(numpy.log(u) / ((i+1)*numpy.log(x)))) for i, u in enumerate(uniform().rvs(n))]
            partition = {(i+2):y for i, y in enumerate(geom_rvs[1:]) if y != 0}

            U = uniform().rvs(size=1)
            residual = int(n - numpy.sum([x*y for x,y in partition.items()]))
            if U < geom.pmf(residual, 1-x, -1) / geom.pmf(0, 1-x, -1):
                keep_going = False
                if residual > 0:
                    partition[1] = residual
            counts += 1

        sample_list.append(partition)
        count_list.append(counts)

    return [sample_list, count_list]




def binary_index_search_helper(sorted_array, value, lower, upper):
    """Binary search of subset of sorted array when element is not necessarily in list.

    Also works for values outside the range, just make sure the array is sorted!

    Arguments:
        sorted_array (array): sorted (!!!!) array of random access values
        value: value to search for within the array.
        lower: lower index of lower bound
        upper: index of upper bound
    """

    midpoint = int(upper - (upper - lower)/2)
    mid_value = sorted_array[midpoint]

    if midpoint <= lower:
        return lower
    if mid_value == value:
        return midpoint
    elif mid_value > value:
        return binary_index_search_helper(sorted_array, value, lower, midpoint)
    else:
        return binary_index_search_helper(sorted_array, value, midpoint, upper)
    
def binary_index_search(sorted_array, value):
    """Binary search of sorted array when element is not necessarily in list.

    Also works for values outside the range, just make sure the array is sorted!

    Arguments:
        sorted_array (array): sorted (!!!!) array of random access values
        value: value to search for within the (entirety of the) array.
    """
    n = len(sorted_array)
    if n == 0:
        return 0
    elif n == 1:
        return 0 if value <= sorted_array[0] else 1
    elif n == 2:
        return 0 if value <= sorted_array[0] else 1 if value <= sorted_array[1] else 2
    else: # n is at least 3
        midpoint = int(n/2)
        lower = 0
        upper = n
        
        return binary_index_search_helper(sorted_array, value, lower, upper)


def array_method_sampling(**kwargs):
    """Generates samples according to Nijenhuis and Wilf's Combinatorial Algorithms, Algorithm RANPAR (Page 75).

    Utilizes Euler's recursion n*p(n) = sum_d sigma(d) p(n-d) to generate partitions.
    """

    size= 1 if 'size' not in kwargs else kwargs['size']
    array = kwargs['array']

    count_list = [1]*size
    sample_list = []

    n = int(kwargs['target'])

    for i in range(size):
        m = int(n)
        partition = defaultdict(int)

        while m > 0:

            j_d_to_weight = {}
            for d in range(1,n+1):
                for j in range(1, n+1):
                    weight = d * array(m - j*d) / (m*array(m))
                    if weight > 0:
                        j_d_to_weight[str(j)+'_'+str(d)] = weight
            #print(numpy.sum(list(j_d_to_weight.values())))

            res = numpy.random.choice(list(j_d_to_weight.keys()), p=list(j_d_to_weight.values()))
            j, d = [int(x) for x in res.split('_')]
            #print(j,d)
            partition[d] += j
            m -= j*d
            #print(m)
        sample_list.append(dict(partition))

    return sample_list, count_list


def table_method_sampling(self, **kwargs):
    """Generates samples according to Nijenhuis and Wilf's Combinatorial Algorithms, using 2D recursion.

    Uses the recursion p(k,n) = p(k-1,n) + p(k, n-k) to generate partitions one part at a time.

    The largest cost is that of creating and storing the table.

    If kwargs has a rows parameter, it will sample from the set of partitions of n into parts 
    of size at most rows.
    """

    size= 1 if 'size' not in kwargs else kwargs['size']
    table = kwargs['table']

    count_list = [1]*size
    sample_list = []
    lower = 0

    for i in range(size):
        n = int(kwargs['target'])
        k = int(n) if 'rows' not in kwargs else int(kwargs['rows'])
        upper = table[k][n]

        part_size = []
        max_size = k
        variate = randint(lower, upper)
        #variate = U.rvs()
        #variate = 27
        #counter = 0
        while n > 0 and max_size > 0 and variate > 0:
            #counter += 1
            column = [table[i][n] for i in range(max_size+1)]
            max_size = 1 + binary_index_search(column, variate)

            part_size.append(max_size)

            #print(column, variate, max_size, n)

            variate = variate - table[max_size-1][n]
            n = n - max_size

        # Fill in remaining 1s
        part_size += [1]*n

        #print(part_size)
        partition = {}
        for part in part_size:
            if part in partition:
                partition[part] += 1
            else:
                partition[part] = 1

        sample_list.append(partition)

    return [sample_list, count_list]



def pdc_recursive_method_sampling(self, **kwargs):
    """Combines the Rejection/Boltzmann and table methods.

    Generates (Z_k+1, ..., Z_n) like in rejection sampling.
    Accepts/Rejects this variate according to PDC
    Samples partitions of m with parts of size at most k using table method above.

    Requires kwargs to contain both a tilting parameter as well as table parameters.
    """


    size= 1 if 'size' not in kwargs else kwargs['size']
    
    table = kwargs['table']
    x = kwargs['tilt']
    rows = kwargs['rows']
    n = kwargs['target']
    
    # row_max = max([table[rows][i]*x**(i) for i in range(n+1)])
    # probs = [table[rows][i]*x**(i)/row_max for i in range(n+1)]

    # Sometimes you have a very large integer and a very small decimal value
    try:
        floating_row = [table[rows][i]*x**(i) for i in range(n+1)]
    except OverflowError as e:
        floating_row = [Decimal(table[rows][i])*Decimal(x)**Decimal(i) for i in range(n+1)]

    row_max = max(floating_row)
    probs = [y/row_max for y in floating_row]



    #print(rows)

    sample_list = []
    count_list = []

    for ii in range(size):
        partition = {}
        counts = 0
        keep_going = True
        while keep_going is True:

            geom_rvs = [int(numpy.floor(numpy.log(u) / ((i+rows+1)*numpy.log(x)))) for i, u in enumerate(uniform().rvs(n-rows))]
            partition = {(i+rows+1):y for i, y in enumerate(geom_rvs) if y != 0}

            U = uniform().rvs(size=1)
            residual = int(n - numpy.sum([x*y for x,y in partition.items()]))
            if residual >= 0 and residual <= n and U < probs[residual]:
                keep_going = False
                if residual > 0:
                    # Do table method sampling with residual with parts <= rows
                    # Do table method sampling with residual with parts <= rows
                    local_kwargs = dict(kwargs)
                    local_kwargs['target'] = residual
                    local_kwargs['rows'] = rows
                    local_kwargs['method'] = 'table_only'
                    local_kwargs['table'] = table
                    local_kwargs['size'] = 1
                    local_partition = self.table_method_sampling(**local_kwargs)
                    #print(local_partition)
                    
                    # update is ok because part sizes are disjoint
                    partition.update(local_partition[0][0])

            counts += 1

        sample_list.append(partition)
        count_list.append(counts)

    return [sample_list, count_list]




