# Why?
This library was created to perform operations like iterating or random sampling of combinatorial structures like integer partition, permutations, set partitions etc.

# What?
Right now only **integer partitions** are implemented, minimally.  One can randomly sample using several different methods.

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

