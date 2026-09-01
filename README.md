# Why?
This library was created to perform operations like iterating or random sampling of combinatorial structures like integer partition, permutations, set partitions etc.

# What?
Right now **integer partitions** and **set partitions** are implemented, minimally.  One can randomly sample either using several different methods.

# Example

    from CombinatorialProbability import IntegerPartition
	ip = IntegerPartition()
	ip.fit(weight=10, make_array=True, make_table=True, make_tilt=True)
	ip.sampling(size=10, method='rejection')

Right now sample returns a tuple, the first element is the sample, the second element is the number of iterations before a successful sample was found, by default it is a list of all 1s if a method is not a rejection method.

Other arguments for method are:
* pdcdsh -- Probabilistic divide-and-conquer deterministic second half
* table_only -- The (tabular) recursive method of Nijenhuis--Wilf
* array_only -- The (array) recursive method of Nijenhuis--Wilf
* pdc-recursive -- Probabilistic divide-and-conquer combined with the table method of Nijenhuis--Wilf

Additional parameters for a given method should be in the form of a dictionary method_params = {} also input to the sample() method.

	ip.sampling(size=10, method='pdcdsh')
	ip.sampling(size=10, method='table_only')
	ip.sampling(size=10, method='array_only')
	ip.sampling(size=10, method='pdc-recursive', method_params={'rows': 3})

# Set partitions

Same shape.  `weight` is the size of the ground set, so this samples partitions of
{1, 2, ..., 10}; a sample is a tuple of blocks, ordered by least element.

    from CombinatorialProbability import SetPartition
	sp = SetPartition()
	sp.fit(weight=10, make_array=True, make_table=True, make_tilt=True)
	sp.sampling(size=10, method='rejection')       # -> [[((1, 4, 7), (2,), (3, 5, 6, 8, 9, 10)), ...], [3, ...]]

Arguments for method are:
* rejection (or boltzmann) -- Boltzmann sampling of SET(SET>=1(Z)), tilted by x*e^x = n
* stam -- Stam's urn algorithm, drawing the number of urns from Dobinski's formula
* table_only -- Unranking a restricted growth string in lexicographic order
* array_only -- The (array) recursive method of Nijenhuis--Wilf, i.e., algorithm RANEQU
* stirling -- By number of blocks, using S(n,k)

The number of blocks can be capped or fixed.  A cap is honoured by the table method and an
exact count by the Stirling method; the other methods raise rather than ignore a restriction
they cannot sample under.

	sp.fit(weight=10, max_blocks=3)
	sp.sampling(size=10, method='table_only')      # at most 3 blocks

	sp.fit(weight=10, blocks=4)
	sp.sampling(size=10, method='stirling')        # exactly 4 blocks

Counting follows the same restriction: `sp.count()` is the Bell number B(n), or S(n,k) with
blocks=k, or a partial row sum with max_blocks=K.  Both generators are iterable, and a set
partition can be converted to its restricted growth string with `sp.to_rgs`, or to the
integer partition of its block sizes with `sp.block_sizes`.