import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal
import scipy
import matplotlib.mlab as mlab

mean = 0
variance = 0.5
sigma = math.sqrt(variance)
N = 10000
ticks = np.linspace(mean-3*sigma, mean+3*sigma, N/10)

# Draw 1d Gaussian Curve
plt.plot(ticks,mlab.normpdf(ticks,mean,sigma))
plt.show()


# Draw 1d Gaussian Curve from random sampling
data = np.random.normal(mean, variance, N)
density = scipy.stats.gaussian_kde(data)
plt.plot(ticks, density(ticks), color='b')
plt.show()


# Draw 2d Gaussian Curve
x, y = np.mgrid[-1.0:1.0:30j, -1.0:1.0:30j] # Arrange (30,30) Grid
xy = np.column_stack([x.flat, y.flat])

# Initialize parameters
mean = np.array([0.0, 0.0])
covariance = np.array([[0.25, 0],
                       [0,    0.25]])
#covariance = np.array([[0.2,0.4],[0,1]])

z = multivariate_normal.pdf(xy, mean=mean, cov=covariance)
z = z.reshape(x.shape) # Z.reshape() -> (30, 30) grid.

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,y,z)
plt.show()

