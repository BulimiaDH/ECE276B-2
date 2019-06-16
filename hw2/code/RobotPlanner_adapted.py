import numpy as np

class RobotPlanner:
    __slots__ = ['boundary', 'blocks']

    def __init__(self, boundary, blocks):
        self.boundary = boundary
        self.blocks = blocks
  
    
    def Valid_Checker(self, newrp):
        """
            To check current position is valid or not.
        """
        # Check if this direction is valid
        if( newrp[0] < self.boundary[0,0] or newrp[0] > self.boundary[0,3] or \
           newrp[1] < self.boundary[0,1] or newrp[1] > self.boundary[0,4] or \
           newrp[2] < self.boundary[0,2] or newrp[2] > self.boundary[0,5] ):
            valid = True
        else:
            valid = False
            return valid
            
        for k in range(self.blocks.shape[0]):
            if( newrp[0] > self.blocks[k,0] and newrp[0] < self.blocks[k,3] and\
               newrp[1] > self.blocks[k,1] and newrp[1] < self.blocks[k,4] and\
               newrp[2] > self.blocks[k,2] and newrp[2] < self.blocks[k,5] ):
                valid = False
          
        return valid
    
    def plan(self, start, goal, method='A_star'):
        # for now greedily move towards the goal
        newrobotpos = np.copy(start)
        
        numofdirs = 26 # number of directions?
        [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
        dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
        dR = np.delete(dR,13,axis=1)
        dR = dR / np.sqrt(np.sum(dR**2,axis=0)) / 2.0 # discrete the environment of robot over [-1,1]x[-1,1]x[-1,1]
        
#        mindisttogoal = np.inf # min distance to goal
        
        if method == 'A_star':  
            newrobotpos = self.A_star(self, start, goal, dR, numofdirs)
        
        return newrobotpos
    

        
    def A_star(self, start, goal, dR, numofdirs, eps=1):
        """
            Weighted A* algorithm.
            Inputs:
                start - start position
                goal - goal
                dR - possible displacements in world frame metrics
                numofdirs - number of directions
                eps - espilon consistancy               
            Outputs:
                
        """
        mindisttogoal = np.inf # min distance to goal
        for k in range(numofdirs):
            newrp = start + dR[:,k]
  
            valid = self.Valid_Checker(self, newrp)
            if valid == False:
                break
  
            # Update newrobotpos
            disttogoal = sum((newrp - goal)**2)
            if( disttogoal < mindisttogoal):
                mindisttogoal = disttogoal
                newrobotpos = newrp
        
        return path, NoM
    
    def LRTA(self, start, goal):
        """
            Learning Real-Time A* algorithm.
        """
        
        pass
    
    def RTAA(self, start, goal):
        """
            Real-Time Adapted A* algorithm. 
        """
        
        pass
    
    def RRT(self, start, goal):
        """
            RRT algorithm.
        """
        
        pass
    
    def RRT_star(self, start, goal):
        """
            RRT* algorithm.
        """
        
        pass
