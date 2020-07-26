
import numpy


def make_p_n_k_table(self, n, k, **kwargs):


    if self.p_n_k_table is None:
        table_columns = numpy.max([n+1, 2])
        table_rows = numpy.max([k+1, 2])
        self.p_n_k_table = [[None for _ in range(table_columns)] for _ in range(table_rows)]

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

        #print('more rows: ', n, k, len(self.p_n_k_table))
        if len(self.p_n_k_table) < (k+1):
            for i in range(k+1 - len(self.p_n_k_table)):
                self.p_n_k_table += [[None]*(n+1)]

    if self.p_n_k_table[k][n] is None:

        for i in range(1, n+1):
            self.p_n_k_table[0][i] = 0
        for i in range(1, k+1):
            self.p_n_k_table[i][0] = 1


        for i in range(1, k+1):
            self.p_n_k_table[i][0] = 1
            for j in range(1, n+1):
                # print('i=', i, ' j=', j)
                # print(self.p_n_k_table)
                if self.p_n_k_table[i][j] is None:
                    if i > j:
                        self.p_n_k_table[i][j] = self.p_n_k_table[j][j]
                    else:
                        # print('i=',i, 'j-i=', j-i, 'p(i,j-i) = ', self.p_n_k_table[i][j-i])
                        # print('i-1=', i-1, 'j-1=', j-1, 'p(i-1,j-1)=', self.p_n_k_table[i-1][j])
                        first_term = 0 if j-i < 0 else self.p_n_k_table[i][j-i]
                        second_term = 0 if (i-1 < 0 or j-1 < 0) else self.p_n_k_table[i-1][j]
                        # print(first_term, second_term)
                        #self.p_n_k_table[i][j] = self.p_n_k_table[i][j-i] + self.p_n_k_table[i-1][j-1]
                        #print(first_term, second_term)
                        self.p_n_k_table[i][j] = first_term + second_term

    #return self.p_n_k_table[k][n]

def p_n_k(self, n, k, **kwargs):

    if not (self.p_n_k_table is not None and len(self.p_n_k_table) >= k+1 and len(self.p_n_k_table[0]) >= n+1):
        self.make_p_n_k_table(n, k, **kwargs)

    return self.p_n_k_table[k][n]


def make_p_of_n_array(self, n, **kwargs):

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

    if not (self.p_of_n_array is not None and len(self.p_of_n_array) >= n+1):
        self.make_p_of_n_array(n, **kwargs)

    return 0 if n < 0 else self.p_of_n_array[n]

