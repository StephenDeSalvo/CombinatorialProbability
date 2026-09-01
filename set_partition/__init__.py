""" @file __init__.py
    @brief Initialization file for SetPartition class.

    This file contains the class definition of SetPartition.
"""


# For combinatorial structures, we implement
from CombinatorialProbability.combinatorics import CombinatorialSequence
from CombinatorialProbability.combinatorics import CombinatorialStructure


class SetPartition(CombinatorialSequence, CombinatorialStructure):
    """Generator for set partitions.

        This class is meant as a generator for set partitions of [n] = {1, 2, ..., n}.  It
        inherits from CombinatorialSequence and CombinatorialStructure.  Several algorithms
        for uniform generation with a fixed size are implemented, and the number of blocks
        can be either fixed or capped.

        This class is not meant to be interpretted as a set partition object itself.  A
        generated partition comes back as a tuple of tuples, blocks ordered by their least
        element -- see the _transforms module for that form and its restricted growth
        string counterpart.

        An intended future feature is to specify restrictions on block sizes, for example,
        partitions with no singleton block, in the same way the number of blocks can be
        restricted now.  That type of property would be specified at the generator level.

    """

    def __init__(self, **kwargs):
        """Initializes the generator for set partitions.

        """

        # Initialize the sub-classes
        CombinatorialSequence.__init__(self, self)
        CombinatorialStructure.__init__(self, self)

        # Initialize the primary properties
        self.bell_array = None
        self.s_n_k_table = None
        self.rgs_tables = None
        self.target = {}
        self.block_sizes_allowed = None

        # Future feature: if there is a restriction on block sizes, it would be specified
        # on the generator object.
        if 'block_sizes' in kwargs:
            self.block_sizes_allowed = kwargs['block_sizes']

    # These are the functions related to the recursive properties.
    from CombinatorialProbability.set_partition._make_table import (
        make_bell_array, bell, make_s_n_k_table, s_n_k, make_rgs_table, rgs_completions)

    # Mimicking the sklearn library, this function "tunes" or "precomputes" as needed for a
    # specified weight or more general target, e.g., set partitions of [n] into k blocks.
    from ._fit import fit

    # A counting function, utilized by CombinatorialSequence.
    from ._partition_function import partition_function

    # The method which generates samples according to a prescribed method
    from ._sampling import sampling, table_method_sampling, stirling_method_sampling
    # Note: these two are member functions because they dynamically allocate a larger table
    # on demand.

    # Utilized by CombinatorialStructure to generate set partitions one at a time.
    from ._iterator_methods import next_object

    # Utility functions to move between the block form and the restricted growth string,
    # and to read off the integer partition of block sizes.
    from ._transforms import to_blocks, to_rgs, canonicalize, block_sizes
