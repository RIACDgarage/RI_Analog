"""
 a random design with right log distribution
"""
import numpy as np

def random_design_function(dmin, d1max, d2max):
    d0 = np.random.rand()*(np.log(d1max) - np.log(dmin)) + np.log(dmin)
    d0 = int(np.exp(d0))
    d1 = np.random.rand()*(np.log(d2max) - np.log(dmin)) + np.log(dmin)
    d1 = int(np.exp(d1))
    return [d0, d1]

"""
for i in range (20):
    print(ranDsn(3, 300, 1000))
"""
