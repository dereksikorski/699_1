import matplotlib.pyplot as plt
from scipy.stats import skewnorm
import numpy as np

omega = 1      # Scale of skew-normal  
alpha = 1  # Skewness
x = 4

z_vals = skewnorm.rvs(a=alpha, loc=x, scale=omega, size=10000) # Find new zs based on skew-normal

plt.hist(z_vals, bins=100)
plt.show()