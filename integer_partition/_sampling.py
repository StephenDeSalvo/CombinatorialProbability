
import numpy
from scipy.stats import geom, uniform, randint

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
        if len(self.p_n_k_table) < n or len(self.p_n_k_table[0]) < n:
            self.make_p_n_k_table(n,n,**kwargs)


        return table_method_sampling(**kwargs)


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
            partition = {(i+1):y for i, y in enumerate([geom.rvs(1-x**i, loc=-1) for i in range(1, n+1) if x**i != 1.0]) if y != 0}
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
    n = self.target['n']
    x = self.x_
    for i in range(size):
        partition = {}
        counts = 0
        keep_going = True
        while keep_going is True:
            partition = {(i+2):y for i, y in enumerate([geom.rvs(1-x**i)-1 for i in range(2, n+1) if x**i != 1.0]) if y != 0}
            U = uniform().rvs(size=1)
            residual = n - numpy.sum([x*y for x,y in partition.items()])
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



def table_method_sampling(**kwargs):
    """Still debugging"""

    size= 1 if 'size' not in kwargs else kwargs['size']
    table = kwargs['table']

    count_list = [1]*size
    sample_list = []
    lower = 1

    for i in range(size):
        n = int(kwargs['target'])
        upper = table[n][n]+1

        part_size = []
        max_size = n
        U = randint(lower, upper)
        variate = U.rvs()
        #variate = 27
        counter = 0
        while n > 0 and max_size > 1 and variate > 0 and counter <= 10:
            counter += 1
            column = [table[i][n] for i in range(max_size+1)]
            max_size = binary_index_search(column, variate)

            part_size.append(max_size)

            print(column, variate, max_size, n)
            
            variate = variate - table[max_size][n]
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



