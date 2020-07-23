


class CombinatorialSequence:
	def __init__(self, object_reference):

		self.object_reference = object_reference


	def count(self, **kwargs):

		return self.object_reference.partition_function(**kwargs)



