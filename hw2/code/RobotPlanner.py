import numpy as np
#import time
from Graph import Graph, Nodes


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2)) 

def diff(lst1, lst2):
    return list(set(lst1).difference(set(lst2)))

class RobotPlanner:
    __slots__ = ['boundary', 'blocks']

    def __init__(self, boundary, blocks):
        self.boundary = boundary
        self.blocks = blocks 
        self.block_planes = []
        for k in range(blocks.shape[0]): 
            
            s1 = blocks[k,:3] # [minx,miny,minz]
            s2 = blocks[k,3:6] # [maxx,maxy,maxz]
            coords = [(s1[0],s1[1],s1[2]),(s1[0],s1[1],s2[2]),(s1[0],s2[1],s1[2]),(s1[0],s2[1],s2[2]),
                      (s2[0],s1[1],s1[2]),(s2[0],s1[1],s2[2]),(s2[0],s2[1],s1[2]),(s2[0],s2[1],s2[2])]
            
            p1c = [coords[0], coords[1], coords[3], coords[2]]
            p1 = self.PlanePara(p1c)
            self.block_planes.append([p1, p1c])
            p2c = [coords[0], coords[4], coords[6], coords[2]]
            p2 = self.PlanePara(p2c)
            self.block_planes.append([p2, p2c])
            p3c = [coords[0], coords[4], coords[5], coords[1]]
            p3 = self.PlanePara(p3c)
            self.block_planes.append([p3, p3c])
            p4c = [coords[7], coords[3], coords[1], coords[5]]
            p4 = self.PlanePara(p4c)
            self.block_planes.append([p4, p4c])
            p5c = [coords[7], coords[6], coords[2], coords[3]]
            p5 = self.PlanePara(p5c)
            self.block_planes.append([p5, p5c])
            p6c = [coords[7], coords[6], coords[4], coords[5]]
            p6 = self.PlanePara(p6c)
            self.block_planes.append([p6, p6c])
            
            
    
    def ComputeNormal(self, p):
        """
            Inputs:
                p - plane, at least including 3 distinct points
            Outputs:
                n - normal vector of the plane
        """
        pt1 = np.array(p[0])
        pt2 = np.array(p[1])
        pt3 = np.array(p[2])
        
        n = np.cross(pt2-pt1, pt3-pt1)
        
        return n
    
    def PlanePara(self, plane):
        """
            Using 4x1 vector to parameterize a 3D plane.
            Inputs:
                plane - at least including 3 distinct points
            Outputs:
                p - [n, d]. n is the normal vector. d is homogeneous coordinates.
        """
        n = self.ComputeNormal(plane)
        d = - n @ np.array(plane[3])
        p = [n, d]
        return p
    
    def plan(self,start,goal):
        # for now greedily move towards the goal
        newrobotpos = np.copy(start)
        
        numofdirs = 26
        [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
        dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
        dR = np.delete(dR,13,axis=1)
        dR = dR / np.sqrt(np.sum(dR**2,axis=0)) / 2.0
    
        mindisttogoal = 1000000
        for k in range(numofdirs):
          newrp = start + dR[:,k] 
          
          # Check if this direction is valid
          if( newrp[0] < self.boundary[0,0] or newrp[0] > self.boundary[0,3] or \
              newrp[1] < self.boundary[0,1] or newrp[1] > self.boundary[0,4] or \
              newrp[2] < self.boundary[0,2] or newrp[2] > self.boundary[0,5] ):
            continue
          
          valid = True
          for k in range(self.blocks.shape[0]):
            if( newrp[0] > self.blocks[k,0] and newrp[0] < self.blocks[k,3] and\
                newrp[1] > self.blocks[k,1] and newrp[1] < self.blocks[k,4] and\
                newrp[2] > self.blocks[k,2] and newrp[2] < self.blocks[k,5] ):
              valid = False
              break
          if not valid:
            break
          
          # Update newrobotpos
          disttogoal = sum((newrp - goal)**2)
          if( disttogoal < mindisttogoal):
            mindisttogoal = disttogoal
            newrobotpos = newrp
        
        return newrobotpos
    
    def valid_checker(self, pts):
        # Check if this direction is valid
        valid2 = True
        for k in range(self.blocks.shape[0]):
            if( pts[0] >= self.blocks[k,0] and pts[0] <= self.blocks[k,3] and\
                pts[1] >= self.blocks[k,1] and pts[1] <= self.blocks[k,4] and\
                pts[2] >= self.blocks[k,2] and pts[2] <= self.blocks[k,5] ):
                return False
            else:
                valid2 = True
        valid1 = ( pts[0] >= self.boundary[0,0] and pts[0] <= self.boundary[0,3] and \
                  pts[1] >= self.boundary[0,1] and pts[1] <= self.boundary[0,4] and \
                  pts[2] >= self.boundary[0,2] and pts[2] <= self.boundary[0,5] )
        
        return valid1 & valid2
    
    def CollisionFree(self, pt1, pt2):
        l = pt1 - pt2
        for p in self.block_planes:
            k = p[0][0] @ l # n @ l
            if k == 0:
                continue
            else:
                t = - (p[0][0] @ pt2 + p[0][1])/k
#                kk = p[0][1]
#                if 0 <= np.abs(t) and np.abs(t) <= 1:
                if 0 <= t and t <= 1:
                    pts = t * pt1 + (1 - t) * pt2
                    pts = np.round(pts, decimals=3)
                    plane_c = np.round(p[1], decimals=3)
                    minx = min([plane_c[0][0], plane_c[2][0]])
                    maxx = max([plane_c[0][0], plane_c[2][0]])
                    miny = min([plane_c[0][1], plane_c[2][1]])
                    maxy = max([plane_c[0][1], plane_c[2][1]])
                    minz = min([plane_c[0][2], plane_c[2][2]])
                    maxz = max([plane_c[0][2], plane_c[2][2]])
                    if( pts[0] >= minx - 1e-2 and pts[0] <= maxx + 1e-2 and\
                       pts[1] >= miny - 1e-2 and pts[1] <= maxy + 1e-2 and\
                       pts[2] >= minz - 1e-2 and pts[2] <= maxz + 1e-2 ):
                        return False
        
        return True


class Search_m(RobotPlanner):
    """
        Search-based method
    """
    def __init__(self, boundary, blocks, scale=0.1):
        RobotPlanner.__init__(self, boundary, blocks)
        # discrete the space into grid
        self.scale = scale
#        self.xrange = np.arange(-1, 1 + self.scale, self.scale)
#        self.yrange = np.arange(-1, 1 + self.scale, self.scale)
#        self.zrange = np.arange(-1, 1 + self.scale, self.scale) # grid
#        [self.dX,self.dY,self.dZ] = np.meshgrid(self.xrange,self.yrange,self.zrange)
#        self.pts = np.vstack((self.dX.flatten(),self.dY.flatten(),self.dZ.flatten()))
#        self.labels={}
    
    def heuristic(self, x1, x2, way='l2'):
        """
            Different heuristic functions
        """
        if way == 'l1':
            return np.linalg.norm(x1-x2, 1)
        if way == 'l2':
            return np.linalg.norm(x1-x2, 2)
        if way == 'inf':
            return np.linalg.norm(x1-x2, np.inf)
        if way == '-inf':
            return np.linalg.norm(x1-x2, -np.inf)
#            return 0.01 * np.max(x1-x2) + 400 * np.min(x1-x2)
        if way == 'custom':
            x = np.abs(x1 - x2)
#            return np.linalg.norm(x)
            return 1 * np.max(x) + 0.4 * np.min(x)
        
    def Goal_test(self, s, goal):
        if sum((s-goal)**2) <= 0.1:
            return True
    
#class LRTA_star(Search_m):
#    """
#        Learning Real-Time A* algorithm.
#    """
#    def __init__(self, boundary, blocks, scale=0.1):
#        Search_m.__init__(self, boundary, blocks, scale)
#        self.H = {} # H is a table of cost estimates indexed by state, recording visited points
#        self.action = np.zeros((3))
#        self.CLOSED = []
#        
#    def plan(self, start, goal):
#        start = np.round(start, decimals=3)
#        if self.Goal_test(start, goal):
#            return start
#        if str(start) not in self.H:
#            self.H[str(start)] = self.heuristic(start, goal)
#            self.CLOSED.append(str(start))
#        
#        
#        # find Children(current_state)
##        [dX,dY,dZ] = np.meshgrid(self.xrange,self.yrange,self.zrange)
#        [dX,dY,dZ] = np.meshgrid([-self.scale,0,self.scale],[-self.scale,0,self.scale],[-self.scale,0,self.scale])
#        dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
#        
#        diff = np.array([[0],[0],[0]]) - dR
#        idx = np.argmin(np.sum(diff**2,axis=0))
#        
##        idx = 13
#        dR = np.round(np.delete(dR,idx,axis=1), decimals=3)
#        dRtmp = list((start.reshape(3,1) + dR).T)
#        
#        # find minimum dist f
#        H_min = np.inf
#        for pt in dRtmp:
##            lst = self.CLOSED
#            v1 = str(pt) in self.CLOSED
#            v2 = self.valid_checker(pt)
#            if (v2) :
#                if not(v1):
#                    h = self.heuristic(pt, goal, way='l1')
#                    self.H[str(pt)] = h
#                    self.CLOSED.append(str(pt))
##                H_tmp = np.round(self.H[str(pt)] + self.cost(start, pt), decimals=2)
#                H_tmp = self.H[str(pt)] + np.linalg.norm(start-pt,2)
#                if H_tmp < H_min:
#                    H_min = H_tmp
#                    best_pt = pt
##        self.H[str(start)] = H_min
#        self.H[str(start)] = np.max([self.H[str(start)], H_min])
#        print(self.H[str(start)])
#        self.action = best_pt
#        
#        return self.action

        
class RTAA_star(Search_m):
    """
        Real-Time Adaptive A* algorithm.
    """
    def __init__(self, boundary, blocks, h, scale=0.1, N = 1):
        Search_m.__init__(self, boundary, blocks, scale)
#        self.action = np.zeros((3))
        self.CLOSED = [] 
        self.OPEN = []
        self.G = {} # record g-value
        self.H = {} # record h-value
        self.heuristics = h # choose which 
        self.scale = scale # the size of grid
        self.N = N # times of expansions
        
    def A_star(self, start, goal, N, eps=1):

        
        # find Children(start)
        [dX,dY,dZ] = np.meshgrid([-self.scale,0,self.scale],[-self.scale,0,self.scale],[-self.scale,0,self.scale])
        dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
        dR = np.round(np.delete(dR,13,axis=1), decimals=1)
            
        # Initialization
        self.OPEN = []
        self.CLOSED = []
        self.OPEN.append(np.round(start, decimals=1))
        self.G = {}
        self.G[str(start)] = 0
        
        for i in range(N):
            # N expansions
            idx = self.Priority_Q(eps, goal) # prioriry quene
            remove_pt = self.OPEN.pop(idx)
            self.CLOSED.append(str(remove_pt))
            children = self.Children(remove_pt, dR)
            
            for pt in children:
                if str(pt) not in self.CLOSED:
                    if str(pt) not in self.G:
                        self.G[str(pt)] = np.inf
                    ftmp = self.G[str(remove_pt)] + np.linalg.norm(pt-remove_pt,2)
                    if (self.G[str(pt)] > ftmp):
                        self.G[str(pt)] = ftmp
                    if self.valid_checker(pt):
                        self.OPEN.append(pt)     
    
    def Priority_Q(self, eps, goal):
        # prioriry quene
        fmin = np.inf
        k = 0
        for pt in self.OPEN:
            if str(pt) not in self.H:
                self.H[str(pt)] = self.heuristic(pt, goal, way=self.heuristics)
            if str(pt) not in self.G:
                self.G[str(pt)] = np.inf
                
            f = eps * self.H[str(pt)] + self.G[str(pt)]
            if f < fmin:
                idx = k
                fmin = f
            k += 1          
        return idx
    
    def Children(self, node, dR):
        # return 26 children
        return list((node.reshape(3,1) + dR).T)
    
    def Goal_test(self, node, goal):
        if sum((node - goal)**2) < 0.1:
            return True
    
    def plan(self, start, goal):
        # execute A_star for expansion
        self.A_star(start, goal, self.N, eps=1)
        
        # j* = argmin{j in OPEN}(gj + hj), find the smallest f-value
        fmin = np.inf
        for pt in self.OPEN:
            if str(pt) not in self.H:
                self.H[str(pt)] = self.heuristic(pt, goal, way=self.heuristics)
            f = self.H[str(pt)] + self.G[str(pt)]
            collifree = self.CollisionFree(start, pt)
            if f < fmin and collifree:
                fmin = f
                best_pt = pt
        
        # update h-values
        for pt in self.CLOSED:
            self.H[str(pt)] = fmin - self.G[str(pt)]
        # move to j, with the smallest f-value
        d = np.linalg.norm(best_pt - start, 2)
        if d > 1:
            best_pt = np.round((best_pt - start)/d * 0.8 + start, decimals=1)
        
        return best_pt
    
    
class RRT_star(RobotPlanner): 
    # Not using this!
    def __init__(self, boundary, blocks, start):
        RobotPlanner.__init__(self, boundary, blocks)
        self.G = Graph(Nodes(start))
#        self.C_free = {}
        Vol_blocks = 0
        for k in range(blocks.shape[0]):
            Vol_blocks += (blocks[k,3] - blocks[k,0]) * (blocks[k,4] - blocks[k,1]) * (blocks[k,5] - blocks[k,2])
        self.Vol_C_free = (boundary[0,3] - boundary[0,0]) * (boundary[0,4] - boundary[0,1]) * (boundary[0,5] - boundary[0,2]) - Vol_blocks # Vol_Cfree
        self.count = 1
        self.best_path = []
        
    def SampleFree(self):
        """
            (Naive version)
            Generate a random point in free space C_free.
        """
        x = np.random.uniform(self.boundary[0,0], self.boundary[0,3])
        y = np.random.uniform(self.boundary[0,1], self.boundary[0,4])
        z = np.random.uniform(self.boundary[0,2], self.boundary[0,5])
        pt_rand = np.array([x, y, z])
        if self.Free(pt_rand):
            return pt_rand
        else:
            pt_rand = self.SampleFree()
            return pt_rand
    
    def Free(self, pts):
        """
            Check if a point is in C_free.
        """
        for k in range(self.blocks.shape[0]):
            if( pts[0] >= self.blocks[k,0] and pts[0] <= self.blocks[k,3] and\
                pts[1] >= self.blocks[k,1] and pts[1] <= self.blocks[k,4] and\
                pts[2] >= self.blocks[k,2] and pts[2] <= self.blocks[k,5] ):
                return False
            else:
                return True
    
    def Steer(self, x, y, eps):
        """
            Return the nearest point z between x and y given radius r, such that z = argmin{z:||z-x||<=eps}||z - y||.
        """
        direction_unit = y - x
        direction_unit = direction_unit/np.linalg.norm(direction_unit)
        z = x + eps * direction_unit
        return z
        
    def r_update(self, k=1.1):
        """
            Update r.
        """
        N = self.G.getNumNodes()
        r = 2 * ((1 + 1/3) * (self.Vol_C_free/(4/3 * np.pi)) * (np.log(N)/N))**(1/3) + 1
        return k * r
    
    def Path(self, start, goal, length=False):
        """
            Reconstruct path based on V and E.
        """
        if length == False:
            return self.G.path(Nodes(start), Nodes(goal))
        else:
            traj = self.G.path(Nodes(start), Nodes(goal))
            if traj == "No Way!":
                return np.inf
            length = 0
            for i in range(1, len(traj)):
                length += np.linalg.norm(traj[i] - traj[i - 1])
            return length
    
    def Cost(self, x, y=None):
        if y == None:
            y = self.G.V.value
        return self.Path(x, y, length=True)
    
#    def plan(self, start, goal):
##        start = np.ro(start, decimals=5)
#        node_nearest = self.G.Nearest(goal)
##        np.random.seed(100)
#        if np.sqrt(sum((node_nearest - goal)**2)) > 1.0: # close enough
#            # if x_nearest is very close to goal, then start to move; Or stay
#            print(np.sqrt(sum((node_nearest - goal)**2)))
#            # RRT*
#            N = 5
#            eps = 20 # could be modified
#            for i in range(N):
#                if np.sqrt(sum((node_nearest - goal)**2)) < 2.0:
#                    x_rand = goal + np.random.randn(3) * 0.1
#                else:
#                    x_rand = self.SampleFree()
#                x_nearest = self.G.Nearest(x_rand)
#                x_new = self.Steer(x_nearest, x_rand, 1)
##                x_new = np.round(x_new, decimals=5)
#
#                if self.CollisionFree(x_nearest, x_new):
#                    # Extend Step
#                    r_star = self.r_update(k=1.1)
#                    X_near = self.G.Near(x_new, min(r_star, eps))
#                    c_min = self.Cost(x_nearest) + np.linalg.norm(x_new - x_nearest)
#                    for x_near in X_near:
#                        if self.CollisionFree(x_new, x_near):
#                            cost_tmp = self.Cost(x_near) + np.linalg.norm(x_near - x_new)
#                            if cost_tmp <= c_min:
#                                x_min = x_near
#                                c_min = cost_tmp
#                    if len(X_near) > 0:
#                        self.G.add_node(Nodes(x_new), Nodes(x_min)) # some issue
#                    # Rewrite Step
#                    for x_near in X_near:
#                        if self.CollisionFree(x_new, x_near):
##                            print(self.Cost(x_new))
#                            if self.Cost(x_new) + np.linalg.norm(x_new - x_near) < self.Cost(x_near):
#                                self.G.modify(Nodes(x_near), Nodes(x_new)) #  change x_near' parent to x_new
#            return start
#        else:
#            # how to move
#            if len(self.best_path) == 0:
#                self.best_path = self.Path(start, node_nearest)
#            
#            # check if it is around the goal
#            if sum((start - goal)**2) < 1:
#                return goal
#            # move to next node, self.count start from 1
#            next_dist = np.linalg.norm(start - self.best_path[self.count])
#            if next_dist >= 1:
#                # if the distance to next node is too far
#                new_pos = (self.best_path[self.count] - start)/np.linalg.norm(self.best_path[self.count] - start) * 0.9 + start
#                return new_pos
#            else:
#                # if the distance to next node is within 1
#                self.count += 1
#                if len(self.best_path) == self.count:
#                    return self.best_path[self.count - 1]
#                else:
#                    return self.best_path[self.count]
    def plan(self, start, goal):
        """
            Plan.
        """
        while True:
            node_nearest = self.G.Nearest(goal)
            if np.linalg.norm(node_nearest - goal) <= 1.0:
                break
            x_rand = self.SampleFree()
            x_nearest = self.G.Nearest(x_rand)
            x_new = self.Steer(x_nearest, x_rand, 0.5)
            if self.CollisionFree(x_nearest, x_new):
                r_star = self.r_update()
                eps = 1
                X_near = self.G.Near(x_new, min([r_star, eps]))
                c_min = self.Cost(x_nearest) + np.linalg.norm(x_new - x_nearest)
                for x_near in X_near:
                    if self.CollisionFree(x_near, x_new):
                        cost_tmp = self.Cost(x_near) + np.linalg.norm(x_near - x_new)
                        if cost_tmp <= c_min:
                            x_min = x_near
                            c_min = cost_tmp
                    if len(X_near) > 0:
                        self.G.add_node(Nodes(x_new), Nodes(x_min))
                for x_near in X_near:
                    if self.CollisionFree(x_new, x_near):
                        if self.Cost(x_new) + np.linalg.norm(x_new - x_near) < self.Cost(x_near):
                            self.G.modify(Nodes(x_near), Nodes(x_new))
        traj = self.Path(start, node_nearest, length=False)
        return traj
    
class RRT_Connect(RRT_star):
    
    def __init__(self, boundary, blocks, start, goal, eps):
        RRT_star.__init__(self, boundary, blocks, start)
        self.Ga = Graph(Nodes(start))
        self.Gb = Graph(Nodes(goal))
        self.start = start
        self.goal = goal
        self.eps = eps # steer radius
        
    def Extend(self, tree, x):
        """
            Extend-step.
        """
        x_nearest = tree.Nearest(x)
        eps = self.eps
        if np.linalg.norm(x_nearest - x) <= eps:
            x_new = x
        else:
            x_new = self.Steer(x_nearest, x, eps)
        if self.CollisionFree(x_nearest, x_new):
#            print("CollisionFree!")
            tree.add_node(Nodes(x_new), Nodes(x_nearest))
            if np.linalg.norm(x_new - x) < 1e-5:
                return "Reached", x_new
            else:
                return "Advanced", x_new
        return "Trapped", x_new
        
    def Connect(self, tree, x):
        """
            Connect-step
        """
        while True:
            status = self.Extend(tree, x)[0]
            if status != "Advanced":
                break
        
        return status
    
    def Path(self, tree, start, goal, length=False):
        """
            Reconstruct path based on V and E.
        """
        if length == False:
            return tree.Path(Nodes(start), Nodes(goal))
        else:
            traj = tree.Path(Nodes(start), Nodes(goal))
            if traj == "No Way!":
                return np.inf
            length = 0
            for i in range(1, len(traj)):
                length += np.linalg.norm(traj[i] - traj[i - 1])
            return length
    
    def plan(self):
        """
            RRT-Connect Plan.
        """
        cts = 0
        while True:

            x_rand = self.SampleFree()
            flag_exd, x_new = self.Extend(self.Ga, x_rand)
            if not (flag_exd == "Trapped"):
                if self.Connect(self.Gb, x_new) == "Reached":
                    traj1 = self.Path(self.Ga, self.Ga.V.value, x_new)
                    traj2 = self.Path(self.Gb, x_new, self.Gb.V.value)
                    traj = traj1[:-1] + traj2
                    return traj
            G = self.Ga
            self.Ga = self.Gb
            self.Gb = G
            cts += 1
            print(cts)
            
    def Output_Nodes(self):
        """
            Output all nodes in the tree
        """
        self.Ga.search(self.Ga.V)
        self.Gb.search(self.Gb.V)
#        return self.Ga.search(self.Ga.V)[1] + self.Gb.search(self.Gb.V)[1]