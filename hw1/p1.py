#!/usr/bin/env python
# coding: utf-8

# This the program for simulation.

# In[23]:


import numpy as np
P = np.array([[1/2, 1/4, 0, 0, 1/4],
              [1/4, 1/2, 1/4, 0, 0],
              [0, 1/4, 1/2, 1/4, 0],
              [0, 0, 1/4, 1/2, 1/4],
              [1/4, 0, 0, 1/4, 1/2]])
P = np.mat(P) # use matrix for power calculation
p_0 = np.array([[25/150, 20/150, 35/150, 24/150, 46/150]]) # prior
np.set_printoptions(precision=5) 
p_100 = p_0 @ (P**100)
print(p_100)

