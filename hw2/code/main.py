import numpy as np
import time
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import RobotPlanner

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))
  

def load_map(fname):
  mapdata = np.loadtxt(fname,dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b'),\
                                    'formats': ('S8','f', 'f', 'f', 'f', 'f', 'f', 'f','f','f')})
  blockIdx = mapdata['type'] == b'block'
  boundary = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view(('<f4',9))
  blocks = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']].view(('<f4',9))
  return boundary, blocks


def draw_map(boundary, blocks, start, goal):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  hb = draw_block_list(ax,blocks)
  hs = ax.plot(start[0:1],start[1:2],start[2:],'ro',markersize=7,markeredgecolor='k')
  hg = ax.plot(goal[0:1],goal[1:2],goal[2:],'go',markersize=7,markeredgecolor='k')  
#  ax.legend(['start'])
#  ax.legend(['goal'])
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_xlim(boundary[0,0],boundary[0,3])
  ax.set_ylim(boundary[0,1],boundary[0,4])
  ax.set_zlim(boundary[0,2],boundary[0,5])  
  return fig, ax, hb, hs, hg

def draw_block_list(ax,blocks):
  v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype='float')
  f = np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[0,1,2,3],[4,5,6,7]])
  clr = blocks[:,6:]/255
  n = blocks.shape[0]
  d = blocks[:,3:6] - blocks[:,:3] 
  vl = np.zeros((8*n,3))
  fl = np.zeros((6*n,4),dtype='int64')
  fcl = np.zeros((6*n,3))
  for k in range(n):
    vl[k*8:(k+1)*8,:] = v * d[k] + blocks[k,:3]
    fl[k*6:(k+1)*6,:] = f + k*8
    fcl[k*6:(k+1)*6,:] = clr[k,:]
  
  if type(ax) is Poly3DCollection:
    ax.set_verts(vl[fl])
  else:
    pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
    pc.set_facecolor(fcl)
    h = ax.add_collection3d(pc)
    return h


def runtest(mapfile, start, goal, n, eps, h, scale, NE, verbose = True):
  # Instantiate a robot planner
  boundary, blocks = load_map(mapfile)
#  RP = RobotPlanner.RobotPlanner(boundary, blocks)
#  RP = RobotPlanner.LRTA_star(boundary, blocks, scale=0.600)
#  RP, flag = RobotPlanner.RTAA_star(boundary, blocks, h, scale, NE), "A_star"
#  RP = RobotPlanner.RRT_star(boundary, blocks, start), "RRT"
  RP, flag = RobotPlanner.RRT_Connect(boundary, blocks, start, goal, eps), "RRT"
  
  np.random.seed(n)
  
  # Initialize a list to record trajectory
  traj = []
  
  # Display the environment
  if verbose:
    fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)  
  
  # Main loop
  if flag == "RRT":
      # if RRT-variant algorithm, wait until RRT completes then move
      # The check has been completed in RRT-Connect (CollisionFree, within the boudary and not moving too fast)
      traj = RP.plan()
      RP.Output_Nodes()
      nodes = RP.Ga.nodes_list + RP.Gb.nodes_list
      if verbose:
          hs[0].set_xdata(traj[-1][0])
          hs[0].set_ydata(traj[-1][1])
          hs[0].set_3d_properties(traj[-1][2])
          fig.canvas.flush_events()
          plt.show()
      length = ComputeLen(traj)
      print('length:',length)
      traj = np.array(traj)
      nodes = np.array(nodes)
      ax.scatter(traj[:,0],traj[:,1],traj[:,2])
      ax.scatter(nodes[:,0],nodes[:,1],nodes[:,2])
      ax.plot(traj[:,0],traj[:,1],traj[:,2])
      return True, len(traj) - 1, traj, length
          
  # else just search and plan to move
  robotpos = np.copy(start)
  traj.append(start)
  numofmoves = 0
  success = True
  while True:
  
    # Call the robot planner
    t0 = tic()
    newrobotpos = RP.plan(robotpos, goal)
    movetime = max(1, np.ceil((tic()-t0)/2.0)) # /2, within 2 sec
    print('move time: %d' % movetime)

    # See if the planner was done on time
#    if movetime > 1:
#      newrobotpos = np.round(robotpos-0.5 + np.random.rand(3), decimals=1)
#      print('Randomly move')
#    else:
#      print('Planned move')
      
    

    # Check if the commanded position is valid
    if sum((newrobotpos - robotpos)**2) > 1:
      print('ERROR: the robot cannot move so fast\n')
      success = False
    if( newrobotpos[0] < boundary[0,0] or newrobotpos[0] > boundary[0,3] or \
        newrobotpos[1] < boundary[0,1] or newrobotpos[1] > boundary[0,4] or \
        newrobotpos[2] < boundary[0,2] or newrobotpos[2] > boundary[0,5] ):
      print('ERROR: out-of-map robot position commanded\n')
      success = False
    for k in range(blocks.shape[0]):
      if( newrobotpos[0] > blocks[k,0] and newrobotpos[0] < blocks[k,3] and\
          newrobotpos[1] > blocks[k,1] and newrobotpos[1] < blocks[k,4] and\
          newrobotpos[2] > blocks[k,2] and newrobotpos[2] < blocks[k,5] ):
        print('ERROR: collision... BOOM, BAAM, BLAAM!!!\n')
        success = False
        break
    if( success is False ):
      break
    
    # Make the move
    robotpos = newrobotpos
    numofmoves += 1
#    traj.append(robotpos)
    print('number of moves:', numofmoves)
    print('current position after move:', traj[-1])
    print('goal', goal)
    
    # if robot is very closed to goal (<=1), just move to it!
    if np.linalg.norm(robotpos-goal) <= 1:
        if RP.CollisionFree(robotpos, goal):
            robotpos = goal
    traj.append(robotpos)

    
    # Update plot
    if verbose:
      hs[0].set_xdata(robotpos[0])
      hs[0].set_ydata(robotpos[1])
      hs[0].set_3d_properties(robotpos[2])
      fig.canvas.flush_events()
      plt.show()
  #     Check if the goal is reached
    if sum((robotpos-goal)**2) <= 0.10:
      break
  
#  if verbose:
#      hs[0].set_xdata(robotpos[0])
#      hs[0].set_ydata(robotpos[1])
#      hs[0].set_3d_properties(robotpos[2])
#      fig.canvas.flush_events()
#      plt.show()
    
  length = ComputeLen(traj)
  print('length:',length)
  traj = np.array(traj)
  ax.scatter(traj[:,0],traj[:,1],traj[:,2])
  ax.plot(traj[:,0],traj[:,1],traj[:,2])
  return success, numofmoves, traj, length

def ComputeLen(traj):
    length = 0
    for i in range(1, len(traj)):
        length += np.linalg.norm(traj[i] - traj[i - 1])
    return length

def test_single_cube(n, eps, h, scale, NE):    
  start = np.array([2.3, 2.3, 1.3])
  goal = np.array([7.0, 7.0, 6.0])
  success, numofmoves, traj, length = runtest('./maps/single_cube.txt', start, goal, n, eps, h, scale, NE, True)
  print('Success: %r'%success)
  print('Number of Moves: %i'%numofmoves)
  return length
  
def test_maze(n, eps, h, scale, NE):
  start = np.array([0.0, 0.0, 1.0])
#  start = np.array([0.0, 0.0, 5.0])
  goal = np.array([12.0, 12.0, 5.0])
  success, numofmoves, traj, length = runtest('./maps/maze.txt', start, goal, n, eps, h, scale, NE, True)
  print('Success: %r'%success)
  print('Number of Moves: %i'%numofmoves)
  return length

def test_window(n, eps, h, scale, NE):
  start = np.array([0.2, -4.9, 0.2])
  goal = np.array([6.0, 18.0, 3.0])
  success, numofmoves, traj, length = runtest('./maps/window.txt', start, goal, n, eps, h, scale, NE, True)
  print('Success: %r'%success)
  print('Number of Moves: %i'%numofmoves)
  return length

def test_tower(n, eps, h, scale, NE):
  start = np.array([2.5, 4.0, 0.5])
  goal = np.array([4.0, 2.5, 19.5])
  success, numofmoves, traj, length = runtest('./maps/tower.txt', start, goal, n, eps, h, scale, NE, True)
  return length

def test_flappy_bird(n, eps, h, scale, NE):
  start = np.array([0.5, 2.5, 5.5])
  goal = np.array([19.0, 2.5, 5.5])
  success, numofmoves, traj, length = runtest('./maps/flappy_bird.txt', start, goal, n, eps, h, scale, NE, True)
  print('Success: %r'%success)
  print('Number of Moves: %i'%numofmoves) 
  return length

def test_room(n, eps, h, scale, NE):
  start = np.array([1.0, 5.0, 1.5])
  goal = np.array([9.0, 7.0, 1.5])
  success, numofmoves, traj, length = runtest('./maps/room.txt', start, goal, n, eps, h, scale, NE, True)
  print('Success: %r'%success)
  print('Number of Moves: %i'%numofmoves)
  return length

def test_monza(n, eps, h, scale, NE):
  start = np.array([0.5, 1.0, 4.9]) 
#  start = np.array([0.5, 1.0, 0.1]) 
  goal = np.array([3.8, 1.0, 0.1])
  success, numofmoves, traj, length = runtest('./maps/monza.txt', start, goal, n, eps, h, scale, NE, True)
  print('Success: %r'%success)
  print('Number of Moves: %i'%numofmoves)
  return length

if __name__=="__main__":
  n = 1 # random seed number
  eps = 1 # steer radius
  h = 'l2' # heuristic function
  scale = 0.5 # the size of grid
  N = 2 # times of expansion in RTAA*
  
  # RTAA*
  length = test_single_cube(n, eps, h, scale, N) # OK
#  length = test_maze(n, eps, h, scale, N) # -inf-norm
#  length = test_flappy_bird(n, eps, h, scale, N) # OK, scale=0.4,shortest,inf,custom
#  length = test_monza(n, eps, h, scale, N) # '-inf'
#  length = test_window(n, eps, h, scale, N) # OK
#  length = test_tower(n, eps, h, scale, N) #OK N=2,inf
#  length = test_room(n, eps, h, scale, N)
     
  #RRT
#  leng = []
#  seed = [1,2,3,4,5]
#  epss = [2.0,1.0,0.7,0.5,0.3]
#  f = open("record.txt", "a")
#  f.write("maze \n")
#  f.close()
#  for eps in epss:
#      for n in seed:
##          length = test_single_cube(n, eps) # OK
##          length = test_maze(n, eps) # -inf-norm
##          length = test_flappy_bird(n, eps) # OK, scale=0.4,shortest,inf,custom
#    #      length = test_monza(n, eps) # '-inf'
##          length = test_window(n, eps) # OK
##          length = test_tower(n, eps) #OK N=2,inf
#          length = test_room(n, eps) #OK N=1,l1/custom
#          leng.append(length)
##          f = open("record.txt", "a")
##          f.write("n = " + str(n) + " " + "eps = " + "\t" + str(eps) + str(length) + '\n')
##          f.close()
#  
#  len_ave = np.mean(leng)
#  print(leng)
#  print(len_ave)








