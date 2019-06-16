# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:50:44 2019

@author: epyir
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

grid = 100
x = np.linspace(-1, 1, grid)
y = np.linspace(-1, 1, grid)
X, Y = np.meshgrid(x, y)
#X = np.vstack([xv.ravel(), yv.ravel()]) # 2-by-N matrix
#X0 = np.vstack([xv.ravel(), yv.ravel()])
#N = X.shape[1]

Z = 0.5 * (X**2 + Y**2)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)