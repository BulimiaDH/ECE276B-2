# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:06:36 2019

@author: epyir
"""

import numpy as np
P = np.array([[0.9, 0.1, 0, 0, 0, 0, 0],
              [0.5, 0, 0.5, 0, 0, 0, 0],
              [0, 0, 0, 0.8, 0, 0.2, 0],
              [0, 0, 0, 0, 0.6, 0, 0.4],
              [0, 0, 0, 0, 0, 1.0, 0],
              [0, 0, 0, 0, 0, 1.0, 0],
              [0, 0.2, 0.4, 0.4, 0, 0, 0]])

R = np.array([-1, -2, -2, -2, 10, 0, 1]).reshape(7,1)
V = np.zeros((7, 1))
gamma = 1.0

for i in range(200):
    V_next = R + P @ (gamma * V)
    V = V_next
print(V)
    
# directly solve:
#V = np.linalg.inv(np.eye(7)-gamma*P) @ R
#print(V)