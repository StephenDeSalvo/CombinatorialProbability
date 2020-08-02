""" @file __init__.py
    @brief Initialization file for IntegerPartition class.

    This file contains the class definition of IntegerPartition.
"""


# For combinatorial structures, we implement 
from CombinatorialProbability.combinatorics import CombinatorialSequence
from CombinatorialProbability.combinatorics import CombinatorialStructure



class IntegerPartition(CombinatorialSequence, CombinatorialStructure):
    """Generator for integer partitions.

        This class is meant as a generator for integer partitions.  It inherits from CombinatorialSequence
        and CombinatorialStructure.  Several algorithms for uniform generation with a fixed size are 
        implemented.

        This class is not meant to be interpretted as an integer partition object itself.  An intended
        future feature is to specify restrictions on part sizes, for example, all odd parts, or all
        even parts, or parts which are perfect powers, or parts in some specified set U.  This type
        of property would be specified at the generator level, whereas the actual partition generated
        would be an output of a member function.

    """
    def __init__(self, **kwargs):
        """Initializes the generator for integer partitions.

        """
        
        # Initialize the sub-classes
        CombinatorialSequence.__init__(self, self)
        CombinatorialStructure.__init__(self, self)

        # Initialize the primary properties
        self.p_of_n_array = None
        self.p_n_k_table = None
        self.target = {}
        self.part_sizes = None

        # Future feature: if there is restriction on part sizes, it would be specified on the generator object.
        if 'part_sizes' in kwargs:
            self.part_sizes = kwargs['part_sizes']


    # These are the functions related to the recursive properties.
    from CombinatorialProbability.integer_partition._make_table import make_p_n_k_table, p_n_k, make_p_of_n_array, p_of_n

    # Mimicking the sklearn library, this function "tunes" or "precomputes" as needed for a specified weight
    # or more general target, e.g., integer partitions of size n with parts of size at most k.
    from ._fit import fit

    # A counting function, utilized by CombinatorialSequence.
    from ._partition_function import partition_function

    # The method which generates samples according to a prescribed method
    from ._sampling import sampling, table_method_sampling, pdc_recursive_method_sampling
    # Note: table_method_sampling is a member function because it dynamically allocates a larger table on demand.

    # Utilized by CombinatorialStructure to generate partitions one at a time.
    from ._iterator_methods import next_object

    # Utility function to quickly map a partition into a more familiar multiset of positive integers.
    from ._transforms import partition_map_to_tuple


