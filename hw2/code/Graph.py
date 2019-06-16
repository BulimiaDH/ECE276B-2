# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:20:11 2019

@author: epyir
"""
import numpy as np
import copy
    
class Nodes(object):
    """
        The node of the tree.
    """
    __slots__ = ['value', 'children']

    def __init__(self, value):
        self.value = np.round(value, decimals=7) # the value of current node
        self.children = [] # a list of children nodes
    
    def Identical(self, node):
        """
            Check if two nodes are the same.
        """
        if np.linalg.norm(self.value - node.value) < 1e-5:
            return True
        else:
            return False
        
        
class Graph(object):
    def __init__(self, start):
        self.V = start
        self.E = [0]
        self.num_nodes = 1
        self.count = 0
    
    def add_node(self, node, parent=None):
        """
            add a new node with value, connected to its parent
        """
        self.node_exist = False
        self.NodeIsExist(self.V, node)
        if self.node_exist:
            print('Error: Node has already existed!')
        else:
            if parent == None:
                # if parent is None, default as root
                root_children = self.V.children
                root_children.append(node)
                self.V.children = root_children
#                print('Add node sucessfully!')

            else:
                # else check if its parent exists or not
                self.node_exist = False
                self.NodeIsExist(self.V, parent)
                if self.node_exist:
                    self.add_node_recursion(parent, node, self.V)
#                    print('Add node sucessfully!')
                else:
                    raise('Error: parent dosen\'t exist!')
                    print('Error: parent dosen\'t exist!')
                    
                # Update number of nodes
                self.num_nodes += self.search(node) + 1    
    
    def add_node_recursion(self, parent, node, tree):
        """
            Recursive function for add_node().
        """
        if parent.Identical(tree):
            tree.children.append(node)
            return
        for child in tree.children:
            if child.Identical(parent):
                child_lst = child.children
                child_lst.append(node)
                child.children = child_lst
                break
            else:
                self.add_node_recursion(parent, node, child)
    
    def remove_node(self, node):
        """
            Reomove node.
        """
        self.node_exist = False
        self.NodeIsExist(self.V, node, search=False, delete=True)
        if not(self.node_exist):
            print('Error: node dosen\'t exist!')
        else:
#            print('Node has been removed!') 
            self.num_nodes -= (self.count + 1) # update number of nodes
            self.count = 0
 
    def modify(self, node, new_parent=None):
        """
            Modify the parent of input node.
        """
        self.node_exist = False
        self.NodeIsExist(self.V, node)
        if not self.node_exist:
            print("Error: Node doesn't exist!")
        else:
            if new_parent == None:
                self.node_exist = False
                self.NodeIsExist(self.V, node, search=False, delete=True)
                root_children = self.V.children
                root_children.append(node)
                self.V.children = root_children
            else:
                self.node_exist = False
                self.NodeIsExist(self.V, new_parent)
                if self.node_exist:
                    # if parent node exists
                    self.node_exist = False
#                    self.NodeIsExist(self.V, node, delete=True)
                    self.remove_node(node)
                    self.add_node_recursion(new_parent, node, self.V)
                else:
                    # if parent node does not exist
                    print("Error: Parent node dosen't exist!")
       
    def search(self, node=None):
        self.nodes_list = []
        self._search(node)
        count = self.count
        self.count = 0
        return count
    
    def _search(self, node=None):
        """
            Traversal this tree from some given node.(DFS)
            Update the number of nodes.
        """
        if node == None:
            # if None, then search from the root
            node = self.V
            self.nodes_list.append(node.value)
            
        for child in node.children:
            self.nodes_list.append(child.value)
            self._search(child)
            self.count += 1
#        print('Search Completed!')
#        print('Number of nodes updated!')
#        print(self.count)
        
    def getNumNodes(self):
        """
            Return number of all nodes
        """
        return self.num_nodes

    def NodeIsExist(self, tree, node, search=False, delete=False):
        """
            Check if node exists in the tree.
            Inputs:
                node - node
                tree - tree structure
                search - if it is true, then return node's parent and children if node exists
                delete - if it is true, then we delete node if it exists. 
        """

#        value = node.value # return Point example
        if node.Identical(self.V):
            self.node_exist = True
        if self.node_exist:
            return 1
        for child in tree.children:
            if child.Identical(node):
                self.node_exist = True
                if search == True:
                    self.search_rlt_parent = tree.value
                    for nchild in child.children:
                        self.search_rlt_children.append(nchild.value)
                if delete == True:
                    tree.children.remove(node)
                break
            else:
                self.NodeIsExist(child, node, search, delete)

  
    def get_edge(self, a, b):
        """
            Return the edge connecting node a & b.
        """
        a = a.value
        b = b.value
        if self.IsConnected(a, b):
            return 
        pass
    
    def Near(self, x, r):
        """
            Return all points in V to x, within distance r.
        """
        self.candidates = []
        if self.distance(self.V, x) < r:
            self.candidates.append(self.V.value)
        self._Near(self.V, x, r)
        
        return self.candidates
        
    def _Near(self, node, x, r):
#        if self.distance(node, x) < r:
#            self.candidates.append(node.value)
            
        for child in node.children:
            if self.distance(child, x) < r:
                self.candidates.append(child.value)
            self._Near(child, x, r)
    
    def Nearest(self, x):
        """
            Return the nearest point to x in V.
            x is nparray.
        """
        self.dist_min = np.inf
        self._Nearest(self.V, x)
        if self.distance(self.V, x) < self.dist_min:
            self.dist_min = self.distance(self.V, x)
            self.node_min = self.V
        
        return self.node_min.value
    
    def _Nearest(self, node, x):
#        if len(node.children) == 0:
#            dist = self.distance(node, x)
#            if dist < self.dist_min:
#                self.dist_min = dist
#                self.node_min = node
#                return
        
        for child in node.children:
            dist = self.distance(child, x)
            if dist < self.dist_min:
                self.dist_min = dist
                self.node_min = child
            self._Nearest(child, x)
    
    def distance(self, node, x):
        """
            distance between node and an array point x.
        """
        return np.linalg.norm(node.value - x)
    
    def Path(self, start, goal):
        """
            Return the shortest path from node start to node goal.
        """
        
        path1 = self.DFS(self.V, start)
        path2 = self.DFS(self.V, goal)
        if path1 == False or path2 == False:
            return "No Way!"
        len1 = len(path1)
        for i in range(len1-1, -1, -1):
            if any((path1[i] == x).all() for x in path2):
                index = [np.array_equal(path1[i],x) for x in path2].index(True)
                path2 = path2[index:]
                path1 = path1[-1:i:-1]
                break
        res = path1 + path2
        return res
    
    def DFS(self, node, goal):
        """
            Depth-first Search.
            node - current node
            goal - goal
        """
        path = []
        path = self._DFS(node, goal, path)
        
        return path
    
    def _DFS(self, node, goal, path):
        path.append(node.value)
        if node.Identical(goal):
            return path
        
        if len(node.children) == 0:
            # if there is no child, return False
            return False 
        
        for child in node.children:
            path_tmp = copy.deepcopy(path)
            res = self._DFS(child, goal, path_tmp)
            if res == False:
                continue
            else:
                return res
        
        return False
        
    
def main():
    # A simple test
    a = np.array([1,2,3])
    b = np.array([4,5,6])
    c = np.array([7,8,9])
    na = Nodes(a)
    nb = Nodes(b)
    nc = Nodes(c)
    G = Graph(na)
    print('Graph generated!')
    G.add_node(nb, na)
    print('Point B added!')
    G.add_node(nc, nb)
    print('Point C added!')
    num = G.getNumNodes()
    print("The number of nodes is", num)
    d = np.array([2,3,4])
    nd = Nodes(d)
    G.add_node(nd, na)
    print('Point D added!')
    print("The number of nodes is", G.getNumNodes())
    
    path = G.path(nc, nd)
    print("Path is", path)
    
    x = np.array([10,8,7])
    node_min = G.Nearest(x)
    print("The nearest point to x = %s is %s !"%(x, node_min))
#    print("The distance is ", dist_min)
    
    r = 8
    v_near = G.Near(x, r)
    print("Within distance %s, vertices are:"%r)
    for v in v_near:
        print(v)
    
    e = np.array([11,2,33])
    f = np.array([99,2,13])
    ne = Nodes(e)
    nf = Nodes(f)
    G_prime = Graph(ne)
    G_prime.add_node(nf, ne)
    
    G.add_node(G_prime.V, nd)
    path = G.path(nf, nc)
    print("Path is", path)
#    G.remove_node(nb)
#    print('Point B removed!')
#    print('Point B exists?', G.NodeIsExist(G.V, nb))
#    num = G.getNumNodes()
#    print("The number of nodes is", num)    
#    e = np.array([11,2,4])
#    ne = Nodes(e)
#    G.add_node(ne, nd)
#    print("Point E added!")
#    print("The number of nodes is", G.getNumNodes())
    
if __name__ == '__main__':
    main()