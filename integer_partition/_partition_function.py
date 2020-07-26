""" @file _partition_function.py
    @brief functions related to counting integer partitions

    The "Partition Function" generally speaking is a normalization function where objects can be weighted
    somewhat arbitrarily, and it normalizes the probability distribution.  For combinatorial objects under
    the uniform distribution it is simply a counting function, which we utilized below. 
"""




def partition_function(self, **kwargs):
    """Returns the number of integer partitions of a given target.

    The target is a property of the class, BUT one can override these properties and simply compute 
    p(n) for any given input n by specifying weight as an optional parameter.

    The kwargs is passed along to these subroutines.

    """

    # Overrides internal value of n stored in target.
    if 'weight' in kwargs:
        weight = int(kwargs['weight'])
        return self.p_of_n(weight, **kwargs)

    # Returns p(n) using internal target when it just has 'n' in it.
    if len(self.target.keys()) == 1 and 'n' in self.target:
        return self.p_of_n(self.target['n'], **kwargs)

    # Returns p(k,n) using internal target when it has both 'n' and 'k' in it.
    if len(self.target.keys()) == 2 and 'n' in self.target and 'k' in self.target:
        return self.p_n_k(self.target['n'], self.target['k'], **kwargs)

