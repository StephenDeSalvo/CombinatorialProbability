


class CombinatorialStructure:
    def __init__(self, object_reference):

        self.object_reference = object_reference

    def initialize(self, combinatorial_object):
        self.combinatorial_object = combinatorial_object

    def __iter__(self):
        return self

    def __next__(self, **kwargs):

        self.combinatorial_object, flag = self.object_reference.next_object(self.combinatorial_object, **kwargs)

        if flag:
            raise StopIteration()

        return self.combinatorial_object



