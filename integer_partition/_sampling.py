
from collections import defaultdict

from random import randint

import numpy
from scipy.stats import geom, uniform


def sampling(self, **kwargs):


    method = 'rejection' if 'method' not in kwargs else kwargs['method']

    # Standard rejection sampling: Sample (Z_1, Z_2, ..., Z_n) --> until sum_i i*Z_i = n
    if method.lower() in ['rejection', 'boltzmann']:

        kwargs['target'] = self.target['n']
        kwargs['tilt'] = self.x_
        #kwargs['distribution'] = geom
        #kwargs['distribution_params'] = {'loc':-1}
        return rejection_sampling(**kwargs)

    # Probabilistic divide-and-conquer: deterministic second half: Sample (Z_2, Z_3, ..., Z_n) --> Until U < P(Z_1 = n-sum_{i\geq 2} i*z_i) / max_j P(Z_1=j)
    elif method.lower() in ['pdcdsh', 'pdc-dsh']:
        kwargs['target'] = self.target['n']
        kwargs['tilt'] = self.x_
        return pdcdsh_sampling(**kwargs)

    elif method.lower() in ['recursive', 'nijenhuis-wilf', 'table_method', 'table_only', 'unrank']:
        kwargs['target'] = self.target['n']
        kwargs['table'] = self.p_n_k_table
        n = self.target['n']
        if self.p_n_k_table is None or len(self.p_n_k_table) < n or len(self.p_n_k_table[0]) < n:
            self.make_p_n_k_table(n,n,**kwargs)

        return table_method_sampling(**kwargs)

    elif method.lower() in ['array_only', 'euler', 'divisors']:

        kwargs['target'] = self.target['n']
        kwargs['array'] = self.p_of_n

        
        if self.p_of_n_table is None or len(self.p_of_n_table) < n:
            self.make_p_of_n_array(n, **kwargs)

        return array_method_sampling(**kwargs)


    elif method.lower().replace('-', ' ').replace('_', ' ') in ['pdc recursive', 'pdc hybrid']:
        kwargs['target'] = self.target['n']

        kwargs['tilt'] = self.x_

        rows = 1 if 'method_params' not in kwargs else 1 if 'rows' not in kwargs['method_params'] else int(kwargs['method_params']['rows'])
        # if 'method_params' in kwargs:
        #     rows = kwargs['method_params']['rows']
        # else:
        #     rows = 1
        #print(rows)

        kwargs['rows'] = rows

        n = self.target['n']


        if self.p_n_k_table is None or len(self.p_n_k_table) < rows or len(self.p_n_k_table[0]) < n:
            self.make_p_n_k_table(n,rows,**kwargs)

        kwargs['table'] = self.p_n_k_table

        return pdc_recursive_method_sampling(**kwargs)




# Note that these sampling methods are not class methods.  
# In the future, they could be implemented separately and more generally based on the arguments in kwargs.
def rejection_sampling(**kwargs):
    size= 1 if 'size' not in kwargs else kwargs['size']

    n = kwargs['target']
    x = kwargs['tilt']
    
    #distribution = kwargs['distribution']
    #distribution_params = {} if 'distribution_params' not in kwargs else kwargs['distribution_params']

    sample_list = []
    count_list = []
    #n = self.target['n']
    #x = self.x_
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

    size= 1 if 'size' not in kwargs else kwargs['size']
    
    # Do not generalize PDCDSH just yet, as there are some subtle pmf issues to be worried about, like max of the pmf.
    # distribuiton = geom

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

            #partition = {(i+2):y for i, y in enumerate([geom.rvs(1-x**i)-1 for i in range(2, n+1) if x**i != 1.0]) if y != 0}
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
    """ Binary search helper function when element is not necessarily in list, returns index of value ASSUMING value IS IN THE RANGE OF VALUES IN THE SORTED ARRAY.

    For my application, this should always be the case.  BUT, it also happens to work for values outside the range.

    JUST MAKE SURE THE ARRAY IS SORTED!!!

    """
    midpoint = int(upper - (upper - lower)/2)
    mid_value = sorted_array[midpoint]
    #print(midpoint, mid_value, lower, upper)
    if midpoint <= lower:
        return lower
    if mid_value == value:
        return midpoint
    elif mid_value > value:
        return binary_index_search_helper(sorted_array, value, lower, midpoint)
    else:
        return binary_index_search_helper(sorted_array, value, midpoint, upper)
    
def binary_index_search(sorted_array, value):
    """ Binary search when element is not necessarily in list, returns index of value ASSUMING value IS IN THE RANGE OF VALUES IN THE SORTED ARRAY.

    For my application, this should always be the case.  BUT, it also happens to work for values outside the range.

    JUST MAKE SURE THE ARRAY IS SORTED!!!

    """
    n = len(sorted_array)
    if n == 0:
        return 0
    elif n == 1:
        return 0 if value <= sorted_array[0] else 1
    elif n == 2:
        return 0 if value <= sorted_array[0] else 1 if value <= sorted_array[1] else 2
    else:
        midpoint = int(n/2)
        lower = 0
        upper = n
        
        return binary_index_search_helper(sorted_array, value, lower, upper)


def array_method_sampling(**kwargs):

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


def table_method_sampling(**kwargs):
    """Finished Debugging"""

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



def pdc_recursive_method_sampling(**kwargs):

    size= 1 if 'size' not in kwargs else kwargs['size']
    
    table = kwargs['table']
    x = kwargs['tilt']
    rows = kwargs['rows']
    n = kwargs['target']
    
    row_max = max([table[rows][i]*x**(i) for i in range(n+1)])
    probs = [table[rows][i]*x**(i)/row_max for i in range(n+1)]

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
                    local_partition = table_method_sampling(**local_kwargs)
                    #print(local_partition)
                    
                    # update is ok because part sizes are disjoint
                    partition.update(local_partition[0][0])

            counts += 1

        sample_list.append(partition)
        count_list.append(counts)

    return [sample_list, count_list]




