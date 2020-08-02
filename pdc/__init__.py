

import desalvo.probability.conditional_distribution as conditional_distribution
import desalvo.probability.distribution_lambda as distribution_lambda

class PDC:
    def __init__(self, **kwargs):
        """Initialies the PDC object similar to the sklearn libraries, and the arguments depend on the pdc_type.

        Examples:
            # Deterministic Second Half
            import numpy
            pdc = PDC(pdc_type='dsh', index=0)
            geometric = [np.]
            PDC.fit()
        """


    def fit(self, distributions, constraints):

        distribution_lambda(constraints, json=True)
        
        conditional_distribution(distributions, )
        return distributions

    def sample(self, n=1, **kwargs):
        """
            Arguments:
                pdc_type (str): This is the type of PDC algorithm to employ.  E.g., trivial/rejection sampling, deterministic second half, self-similar, table hybrid, etc.
        """
        return n
