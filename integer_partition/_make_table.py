""" @file _make_table.py
    @brief Makes the tables or arrays associated with a recursion.

    There are two main recursions for integer partitions:
    1.  1D:  Euler's Pentagonal Theorem.
    2.  2D:  p(k,n) = p(k-1, n) + p(k,n-k)

    By creating an array/table, one can quickly compute p(n) using only integer arithmetic.
    In Python integer arithmetic is arbitrary precision (!!!), so as long as you have memory 
    you can extend the table as large as you like.
"""

# Only used for basic operations like max and sqrt
import numpy


def make_p_n_k_table(self, n, k, **kwargs):
    """Computes and stores p(k,n) into self.p_n_k_table.

    Arguments:
        n (integer): the number of columns
        k (integer): the number of rows
        kwargs (dict): optional parameters

    The table is actually (n+1)*(k+1), since the index 0 is included and the recursion is
    performed up until the value of p(k,n).

    This function is particularly elegant because it first checks to see if the table is 
    currently of the appropriate size, and if so exits in O(1) time.  Otherwise, it extends
    the table in a natural manner.
    """

    # If table does not yet exist, initialize it with None's and apply boundary conditions.
    if self.p_n_k_table is None:
        
        table_columns = numpy.max([n+1, 2])
        table_rows = numpy.max([k+1, 2])

        self.p_n_k_table = [[None for _ in range(table_columns)] for _ in range(table_rows)]

        # Boundary conditions
        self.p_n_k_table[0][0] = 1

        for i in range(1, n+1):
            self.p_n_k_table[0][i] = 0

        for i in range(1, k+1):
            self.p_n_k_table[i][0] = 1


    else:
        # Extend table with more columns
        if len(self.p_n_k_table[0]) < (n+1):
            existing_rows = len(self.p_n_k_table)
            for i in range(existing_rows):
                self.p_n_k_table[i] += [None]*(n+1-len(self.p_n_k_table[i]))

        # extend table with more rows
        if len(self.p_n_k_table) < (k+1):
            for i in range(k+1 - len(self.p_n_k_table)):
                self.p_n_k_table += [[None]*(n+1)]

    # If the value in the table has not yet been computed, then ...
    if self.p_n_k_table[k][n] is None:

        # Boundary conditions
        for i in range(1, n+1):
            self.p_n_k_table[0][i] = 0
        for i in range(1, k+1):
            self.p_n_k_table[i][0] = 1

        # Apply the recursion to those elements which have not yet been computed (i.e., are None)
        for i in range(1, k+1):
            self.p_n_k_table[i][0] = 1
            for j in range(1, n+1):
                if self.p_n_k_table[i][j] is None:
                    if i > j:
                        self.p_n_k_table[i][j] = self.p_n_k_table[j][j]
                    else:
                        first_term = 0 if j-i < 0 else self.p_n_k_table[i][j-i]
                        second_term = 0 if (i-1 < 0 or j-1 < 0) else self.p_n_k_table[i-1][j]

                        self.p_n_k_table[i][j] = first_term + second_term



def p_n_k(self, n, k, **kwargs):
    """Returns the value of p(k,n), extending the table in self.p_n_k_table if necessary.

    Arguments:
        n (integer): the number of columns
        k (integer): the number of rows
        kwargs (dict): optional parameters
    """

    if not (self.p_n_k_table is not None and len(self.p_n_k_table) >= k+1 and len(self.p_n_k_table[0]) >= n+1):
        self.make_p_n_k_table(n, k, **kwargs)

    return self.p_n_k_table[k][n]


def make_p_of_n_array(self, n, **kwargs):
    """Computes and stores p(1), ..., p(n) into self.p_of_n_array.  

    Arguments:
        n (integer): the number of elements
        kwargs (dict): optional parameters

    This function is particularly elegant because it first checks to see if the array is 
    currently of the appropriate size, and if so exits in O(1) time.  Otherwise, it extends
    the array in a natural manner, skipping over those elements already computed.

    If 'sequence' is specified, it will compute p(n) according to a formula or recursion.
    """

    if 'sequence' in kwargs:
        method = kwargs['sequence']
    else:
        if n <= 3600:
            method = 'euler'
        else:
            method = 'hardy-ramanujan'

    if method.lower() == 'euler': #'euler' in kwargs and kwargs['euler'] is True:

        if self.p_of_n_array is None:
            table_size = numpy.max([n+1, 2])
            self.p_of_n_array = [None]*(table_size)
            self.p_of_n_array[0] = 1
            self.p_of_n_array[1] = 1
        else:
            if len(self.p_of_n_array) < (n+1):
                self.p_of_n_array += [None]*(n+1-len(self.p_of_n_array))

        if self.p_of_n_array[n] is not None:
            return self.p_of_n_array[n]

        #print('n = ', n)
        pn = 0

        euler_formula_lower = int(-(numpy.sqrt(24*n+1)-1)/6)
        euler_formula_upper = int((numpy.sqrt(24*n+1)+1)/6)
        for i in range(euler_formula_lower, euler_formula_upper+1):

            index = int(n - i*(3*i-1)/2)
            #print(index)
            if index < n:
                if self.p_of_n_array[index] is None:
                    self.p_of_n_array[index] = self.make_p_of_n_array(index)

                if i % 2 == 0:
                    pn -= self.p_of_n_array[index]
                else:
                    pn += self.p_of_n_array[index]

        self.p_of_n_array[n] = pn
        return pn




    elif method.lower() in ['hardy-ramanujan', 'hr', 'hardy ramanujan', 'hardy and ramanujan']:
        
        if n <= 3600:
            return self.partition_function(weight=n)
        else:
            return 'n > 3600 is not currently supported'


def p_of_n(self, n, **kwargs):
    """Returns the value of p(n), extending the array in self.p_of_n_array if necessary.

    Arguments:
        n (integer): the number of columns
        kwargs (dict): optional parameters
    """

    # Why not, not why.
    if not (self.p_of_n_array is not None and len(self.p_of_n_array) >= n+1):
        self.make_p_of_n_array(n, **kwargs)

    return 0 if n < 0 else self.p_of_n_array[n]

