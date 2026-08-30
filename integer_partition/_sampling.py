""" @file _sampling.py
    @brief Functions related to random sampling of integer partitions.

    A collection of functions which generate random integer partitions each according to a different
    method.  The sampling function is the one to call with method='rejection' specified, for example.

"""


# Cleaner code to just do my_dict[element] += 1 rather than check if element is in my_dict each time
from collections import defaultdict

# Exact on arbitrary-precision ints, unlike anything that goes through floats.
from bisect import bisect_left

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




def array_method_sampling(**kwargs):
    """Generates samples according to Nijenhuis and Wilf's Combinatorial Algorithms, Algorithm RANPAR (Page 75).

    Utilizes Euler's recursion n*p(n) = sum_d sigma(d) p(n-d) to generate partitions.

    Euler's recursion is m*p(m) = sum over d>=1, j>=1 of d*p(m - j*d), so the move
    "add j parts of size d" carries weight d*p(m - j*d) against a total of exactly
    m*p(m).  Picking a move is therefore a walk along the cumulative weights, done
    in whole numbers: p(m) is an arbitrary-precision integer, and turning these
    weights into probabilities would put a huge integer over a huge integer and
    round the answer.

    Only the pairs with j*d <= m carry any weight, and there are about m*log(m) of
    those, not the n^2 that a j and d both running to n would visit.
    """

    size = 1 if 'size' not in kwargs else kwargs['size']
    array = kwargs['array']

    count_list = [1]*size
    sample_list = []

    n = int(kwargs['target'])

    # p(0..n) once up front. This used to be a bound-method call, with kwargs, in
    # the innermost loop -- so it ran n^2 times per part chosen, per sample.
    p = [array(k) for k in range(n + 1)]

    for _ in range(size):
        m = n
        partition = defaultdict(int)

        while m > 0:
            variate = randint(1, m * p[m])
            cumulative = 0
            chosen = None
            for d in range(1, m + 1):
                # jd = j*d walks d, 2d, 3d, ... so j*d <= m always holds and each
                # (j, d) is visited exactly once.
                for jd in range(d, m + 1, d):
                    cumulative += d * p[m - jd]
                    if cumulative >= variate:
                        chosen = (jd // d, d)
                        break
                if chosen is not None:
                    break

            j, d = chosen
            partition[d] += j
            m -= j * d

        sample_list.append(dict(partition))

    return sample_list, count_list


def table_method_sampling(self, **kwargs):
    """Generates samples according to Nijenhuis and Wilf's Combinatorial Algorithms, using 2D recursion.

    Uses the recursion p(k,n) = p(k-1,n) + p(k, n-k) to generate partitions one part at a time.

    The largest cost is that of creating and storing the table.

    If kwargs has a rows parameter, it will sample from the set of partitions of n into parts 
    of size at most rows.

    Numbering.  The partitions of n with parts of size at most k occupy positions
    1..table[k][n], ordered by largest part.  Exactly table[j][n] - table[j-1][n] of
    them have largest part equal to j, so those sit at positions
    table[j-1][n]+1 .. table[j][n].  Decoding a variate is therefore: find the least j
    with variate <= table[j][n], emit j, drop the partitions ranked below it, and carry
    on with what is left of n and parts now capped at j.
    """

    size = 1 if 'size' not in kwargs else kwargs['size']
    table = kwargs['table']

    count_list = [1]*size
    sample_list = []

    target = int(kwargs['target'])
    row_cap = target if 'rows' not in kwargs else int(kwargs['rows'])

    for _ in range(size):
        n = target
        # A part can never exceed what is left to spend, whatever the cap says.
        k = min(row_cap, n)
        part_size = []

        # random.randint is inclusive at BOTH ends. The valid ranks are 1..table[k][n];
        # rank 0, or table[k][n]+1, decodes to a part bigger than n.
        variate = randint(1, table[k][n]) if n > 0 else 0

        while n > 0:
            if k <= 1:
                part_size += [1]*n
                break
            column = [table[j][n] for j in range(k+1)]
            # Least j with variate <= column[j]. column[0] is 0 whenever n > 0 and
            # variate >= 1, so j >= 1; column[k] >= variate, so j <= k <= n. The part
            # can therefore never exceed the cap nor the weight remaining.
            j = bisect_left(column, variate)
            part_size.append(j)
            variate -= table[j-1][n]
            n -= j
            k = min(j, n)

        partition = {}
        for part in part_size:
            partition[part] = partition.get(part, 0) + 1

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

                    # The two halves are disjoint by construction: the geometric
                    # variates above only produce parts > rows, and the table
                    # method only parts <= rows. Add rather than update anyway —
                    # dict.update() silently REPLACES a shared key, so when the
                    # table method used to return an oversized part it destroyed
                    # the geometric multiplicity and the sample came out light.
                    for part, multiplicity in local_partition[0][0].items():
                        partition[part] = partition.get(part, 0) + multiplicity

            counts += 1

        sample_list.append(partition)
        count_list.append(counts)

    return [sample_list, count_list]




