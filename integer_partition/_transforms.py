""" @file _transforms.py
    @brief Contains useful functions relevant to transforming an integer partition.

    This file came out of a need to perform standard operations on integer partitions.
"""



def partition_map_to_tuple(self, partition_map):

    partition_tuple = []
    for part_size, multiplicity in sorted(partition_map.items(), key=lambda pair: -pair[0]):

        partition_tuple += [part_size]*multiplicity

    return tuple(partition_tuple)

