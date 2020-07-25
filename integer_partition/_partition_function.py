

def partition_function(self, **kwargs):

    # Overrides internal value of n stored in target.
    if 'weight' in kwargs:
        weight = int(kwargs['weight'])
        return self.p_of_n(weight, **kwargs)

    if len(self.target.keys()) == 1 and 'n' in self.target:
        return self.p_of_n(self.target['n'], **kwargs)


    if len(self.target.keys()) == 2 and 'n' in self.target and 'k' in self.target:
        return self.p_n_k(self.target['n'], self.target['k'], **kwargs)

