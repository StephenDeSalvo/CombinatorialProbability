""" @file _partition_function.py
    @brief functions related to counting set partitions

    The "Partition Function" generally speaking is a normalization function where objects can be
    weighted somewhat arbitrarily, and it normalizes the probability distribution.  For combinatorial
    objects under the uniform distribution it is simply a counting function, which we utilized below.

    For set partitions it is the Bell number B(n), or the Stirling number of the second kind S(n,k)
    when the number of blocks is fixed, or a partial row sum of those when the number of blocks is
    only capped.
"""


def partition_function(self, **kwargs):
    """Returns the number of set partitions of a given target.

    The target is a property of the class, BUT one can override these properties and simply compute
    B(n) for any given input n by specifying weight as an optional parameter.  Adding blocks=k gives
    S(n,k), and max_blocks=K gives S(n,1) + ... + S(n,K).

    The kwargs is passed along to these subroutines.
    """

    # Overrides the internal value of n stored in target.
    n = None
    for key in ('weight', 'n', 'elements'):
        if key in kwargs:
            n = int(kwargs[key])
            break

    # The table builders take n and k positionally, and the caller's kwargs can carry keys
    # of the same name -- weight=8 arrives as n -- so nothing is forwarded to them here.
    if n is not None:
        if 'blocks' in kwargs:
            return self.s_n_k(n, int(kwargs['blocks']))
        if 'max_blocks' in kwargs:
            return self.rgs_completions(n, int(kwargs['max_blocks']))
        return self.bell(n)

    n = self.target['n']

    # Exactly k blocks.
    if 'k' in self.target:
        return self.s_n_k(n, self.target['k'])

    # At most K blocks.
    if 'max_blocks' in self.target:
        return self.rgs_completions(n, self.target['max_blocks'])

    return self.bell(n)
