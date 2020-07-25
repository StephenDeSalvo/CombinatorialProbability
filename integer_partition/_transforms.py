

def partition_map_to_tuple(self, partition_map):

    partition_tuple = []
    for part_size, multiplicity in sorted(partition_map.items(), key=lambda pair: -pair[0]):

        partition_tuple += [part_size]*multiplicity

    return tuple(partition_tuple)

