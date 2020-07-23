

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', ''))


from desalvo.integer_partition import integer_partition

ip = integer_partition()

ip.fit(size=1000, components=100, sequence='euler', iterator='bidirectional', sampler='PDCDSH')

#print(ip.p_of_n_table)
#print(ip.partition_function())
print(ip.count())

ip.fit(size=5, components=3)
print(ip.count())


ip.fit(size=5)
print(ip.count())

for i in ip:
    print(i)

#samples = ip.sample()

# for i in range(15):
# 	print(ip.count(size=i))

# print(ip.count(size=100))
# print(ip.count(size=30))

# print(ip.p_of_n_table)

# #print(ip.count(size=10000))

# print(ip.count(size=10, components=10))
# print(ip.p_n_k_table)

# integer_partition_fit = ip.fit(size=n, cache=True, iterator='random_access', method='euler')




