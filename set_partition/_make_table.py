""" @file _make_table.py
    @brief Makes the arrays and tables associated with the set partition recursions.

    There are three recursions worth storing:

    1.  1D:  the Bell numbers, B(m) = sum_j C(m-1, j) B(j).  This is the recursion you get
        by asking how big the block containing element 1 is, and it is what the array
        method samples from.

    2.  2D:  the Stirling numbers of the second kind, S(n,k) = k*S(n-1,k) + S(n-1,k-1),
        counting the partitions of [n] into exactly k blocks.  Element n either joins one
        of the k blocks or opens the k-th itself.

    3.  2D:  the number of ways to finish a restricted growth string.  With j positions
        left to fill and m blocks already open, and a cap of K blocks overall,

            C(0, m) = 1,    C(j, m) = m*C(j-1, m) + [m < K] * C(j-1, m+1).

        C(n, 0) is then the number of set partitions of [n] into at most K blocks, and the
        table is exactly what is needed to unrank in lexicographic order on the RGS.

    Everything here is in Python integers, which are arbitrary precision, so the counts are
    exact however large they get -- B(100) is already a 116 digit number, well past what a
    float can hold, and the samplers rely on these being exact.
"""

from math import comb


def make_bell_array(self, n, **kwargs):
    """Computes and stores B(0), ..., B(n) into self.bell_array.

    Arguments:
        n (integer): the largest Bell number required
        kwargs (dict): optional parameters, unused, accepted so callers can pass theirs on

    Returns in O(1) if the array already reaches n, and otherwise extends it from wherever
    it stopped -- the values already computed are never recomputed.
    """

    if self.bell_array is None:
        self.bell_array = [1]                      # B(0) = 1, the empty partition

    for m in range(len(self.bell_array), n+1):
        self.bell_array.append(
            sum(comb(m-1, j)*self.bell_array[j] for j in range(m)))


def bell(self, n, **kwargs):
    """Returns B(n), extending self.bell_array if necessary."""

    if n < 0:
        return 0

    if self.bell_array is None or len(self.bell_array) < n+1:
        self.make_bell_array(n, **kwargs)

    return self.bell_array[n]


def make_s_n_k_table(self, n, k, **kwargs):
    """Computes and stores S(n,k) into self.s_n_k_table[k][n].

    Arguments:
        n (integer): the number of columns, i.e., the size of the ground set
        k (integer): the number of rows, i.e., the largest block count required
        kwargs (dict): optional parameters, unused, accepted so callers can pass theirs on

    Indexed [k][n] to match the p(k,n) table of the integer partition package.  Rows are
    kept at a uniform length, which is what makes the growth cheap: extending columns needs
    row k-1 only one column back, and a new row needs the row below it and nothing else.
    """

    if self.s_n_k_table is None:
        # Row k = 0: S(0,0) = 1 and S(n,0) = 0 for n >= 1.
        self.s_n_k_table = [[1] + [0]*max(n, 1)]

    columns = len(self.s_n_k_table[0]) - 1

    # Extend every existing row with more columns, lowest k first so that row k-1 has
    # already reached the column that row k needs.
    if n > columns:
        for row_index, row in enumerate(self.s_n_k_table):
            for column in range(columns+1, n+1):
                if row_index == 0:
                    row.append(0)
                else:
                    row.append(row_index*row[column-1]
                               + self.s_n_k_table[row_index-1][column-1])
        columns = n

    # Add whole new rows, each computed from the one below it.
    while len(self.s_n_k_table) <= k:
        row_index = len(self.s_n_k_table)
        below = self.s_n_k_table[row_index-1]
        row = [0]*(columns+1)                       # S(0,k) = 0 for k >= 1
        for column in range(1, columns+1):
            row[column] = row_index*row[column-1] + below[column-1]
        self.s_n_k_table.append(row)


def s_n_k(self, n, k, **kwargs):
    """Returns S(n,k), extending the table in self.s_n_k_table if necessary."""

    if n < 0 or k < 0:
        return 0

    if not (self.s_n_k_table is not None
            and len(self.s_n_k_table) >= k+1
            and len(self.s_n_k_table[0]) >= n+1):
        self.make_s_n_k_table(n, k, **kwargs)

    return self.s_n_k_table[k][n]


def make_rgs_table(self, n, max_blocks=None, **kwargs):
    """Computes and returns the RGS completion table for [n] with at most max_blocks blocks.

    Arguments:
        n (integer): the size of the ground set
        max_blocks (integer): cap on the number of blocks, by default n (i.e., no cap)
        kwargs (dict): optional parameters, unused, accepted so callers can pass theirs on

    table[j][m] is the number of ways to fill the last j positions of the string given that
    m blocks have been opened so far.  Tables are cached per cap in self.rgs_tables, since
    a run with a cap of 3 and a run with no cap need genuinely different tables.

    A cap above n is the same as no cap at all, so it is normalized down to n and shares
    that table rather than building a second, larger copy of it.
    """

    cap = n if max_blocks is None else min(int(max_blocks), n)
    cap = max(cap, 0)

    if self.rgs_tables is None:
        self.rgs_tables = {}

    cached = self.rgs_tables.get(cap)
    if cached is not None and len(cached) >= n+1:
        return cached

    table = [[1]*(cap+1)]                           # j = 0: one way, the empty suffix
    for j in range(1, n+1):
        previous = table[j-1]
        table.append([m*previous[m] + (previous[m+1] if m < cap else 0)
                      for m in range(cap+1)])

    self.rgs_tables[cap] = table
    return table


def rgs_completions(self, n, max_blocks=None, **kwargs):
    """Returns the number of set partitions of [n] into at most max_blocks blocks.

    This is table[n][0] of the RGS table: n positions left to fill, no blocks open yet.
    With no cap it is B(n); with a cap of K it is S(n,1) + ... + S(n,K).
    """

    return self.make_rgs_table(n, max_blocks, **kwargs)[n][0]
