# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:32:38 2019

@author: epyir
"""

import numpy as np
import os.path
from p4_utils import load_data, plot_graph, save_results

# an implementation of Label Correcting Algorithm(LCA)

def LCA(n,s,t,C):
    """
        Inputs:
            n - # nodes
            s - start node
            t - ternimate node
            C - cost matrix with c[i, j] = cost of the path from i to j
        Outputs:
            path - the optimal path from s to t
            cost - the optimal cost-to-go values 
    """
    
    OPEN = []
    g_nodes = np.ones((n,)) * np.inf
    g_nodes[s] = 0
    parent = {}
    OPEN.append(s)
    while len(OPEN) is not 0:
        i = OPEN[0]
        del OPEN[0] # always remove the first/the last entry in OPEN
        children = np.where(C[i, :] < np.inf)[0]
        
        for j in children:
            if g_nodes[i] + C[i, j] < g_nodes[j] and g_nodes[i] + C[i, j] < g_nodes[t]:
                g_nodes[j] = g_nodes[i] + C[i, j]
                parent[str(j)] = i
#                if j == 108:
#                    print(i)
                if j is not t:
                    OPEN.append(j)

    path = FindParents(parent, s, t)
    cost = ComputeCost(path[::-1], C)
    return path, cost

def FindParents(dic, s, t):
    """
        Inputs:
            dic - dic is the dictionary recording the parent of each node
            s - start node
            t - ternimate node
        Output:
            path - the optimal path
    """
    key = str(t)
    path = [t]
    while key != str(s):
        parent = dic[key]
        path.append(parent)
        key = str(parent)
    path[0] = path[0].reshape(1)[0]
    path[-1] = path[-1].reshape(1)[0]
    return path[::-1]

def ComputeCost(path, C):
    """
        Input:
            path - the optimal path
        Output:
            cost - the cost at each step
    """
    cost = [0]
    
    for cts, i in enumerate(path):
        if cts != len(path) - 1:
            cost.append(cost[cts] + C[path[cts + 1], i])
        else:
            break
        
    return cost[::-1]
    
if __name__=="__main__":
    input_file = './data/problem5.npz'
    file_name = os.path.splitext(input_file)[0]
      
    # Load data 
    n,s,t,C = load_data(input_file)
#    np.fill_diagonal(C, 0)
    
    # Generate results
    path, cost = LCA(n,s,t,C)
      
    # Visualize (requires: pip install graphviz --user)
#    path = [s,t]
    plot_graph(C,path,file_name)
      
    # Save the results
    save_results(path,cost,file_name+"_results.txt")