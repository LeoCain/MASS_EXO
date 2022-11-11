"""
I just used this to test random bits of code in isolation
"""

import matplotlib.pyplot as plt
import os
import numpy as np

thing = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
print(thing)
path = "{}/Data/testcsv.csv".format(os.path.dirname(__file__))
np.savetxt(path, thing.T, delimiter=",")

my_data = np.genfromtxt(path, delimiter=',')
print(my_data)