

class joint_distribution:
    def __init__(self, distribution_list, **kwargs):

        # Ordered list of distributions.
        self.distribution_list = distribution_list


class distribution_lambda:
    def __init__(self, **kwargs):
        assert('type' in kwargs, 'type not in kwargs')
        lambda_type = kwargs['type']

        if lambda_type == 'sum':
            assert('target' in kwargs, 'when type = sum, argument target must be in kwargs')
            target = kwargs['target']


class conditional_distribution:
    def __init__(self, distribution, constraint):
        """A conditional distribution (A,B,...| lambda(A,B,...; k,l,...))

        For example, we might have ( (X1, X2, ..., Xn) | sum_i X_i = k).
        Here (X1, X2, ..., Xn) is a joint_distribution object
        sum_i X_i = k is a distribution_lambda.
        """

        self.distribution = distribution
        self.constraint = constraint

