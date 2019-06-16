# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:39:36 2019

@author: epyir
"""

"""
==================================
Inverted pendulum animation class
==================================

Adapted from the double pendulum problem animation.
https://matplotlib.org/examples/animation/double_pendulum_animated.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
from scipy.stats import multivariate_normal
import pickle
from scipy import interpolate
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

class EnvAnimate:

    '''
    Initialize Inverted Pendulum
    '''
    def __init__(self, dt=0.5):

        # Change this to match your discretization
#        self.dt = 0.05
        self.dt = dt
        self.t = np.arange(0.0, 100+self.dt, self.dt)

        # Random trajectory for example
        self.theta = np.linspace(-np.pi, np.pi, self.t.shape[0])
        self.x1 = np.sin(self.theta)
        self.y1 = np.cos(self.theta)
        self.u = np.zeros(self.t.shape[0])

#        self.fig = plt.figure()
#        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
#        self.ax.grid()
#        self.ax.axis('equal')
#        plt.axis([-2, 2, -2, 2])
#        
#        self.line, = self.ax.plot([],[], 'o-', lw=2)
#        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
#        self.time_text = self.ax.text(0.05, 0.8, '', transform=self.ax.transAxes)


    def new_data(self, theta, u):
        '''
            Provide new rollout theta values to reanimate
        '''
        self.theta = theta # theta should be an array with shape(t.shape[0],)
        self.x1 = np.sin(theta)
        self.y1 = np.cos(theta) # this is what's gonna show in xy-axis.
#        self.u = np.zeros(self.t.shape[0])
        self.u = u

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2,-2, 2])
        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)

    def init(self):
        self.line.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.time_text

    def _update(self, i):
        thisx = [0, self.x1[i]]
        thisy = [0, self.y1[i]]
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (self.t[i], self.theta[i], self.u[i]))
        return self.line, self.time_text

    def start(self):
        """
        Arguments of FuncAnimation(fig,func,frames,interval,blit,init_func,repeat)
            fig - current figure
            func - The function to call at each time. Determined by theta.
            frames - # of frames
            interval - the delay between frames in milliseconds.
            bilt - 
            init_func - initialize. Can be replaced by self.new_data
            repeat - repeat or not.
        """
        print('Starting Animation')
        print()
        # Set up plot to call animate() function periodically
        self.ani = FuncAnimation(self.fig, self._update, frames=range(len(self.x1)), interval=25, blit=True, init_func=self.init, repeat=False)
        self.ani.save('double_pendulum.mp4', fps=10)
        plt.show()

class IP:
    """
    Environment for Inverted Pendulum.
    """
    def __init__(self, dx1, dx2, du, max_speed=4, max_torque=2, dt=0.5):
        """
        g = 10
        m = 1
        l = 1
        Then a = 3*g/(2*l) = 15
        Damping b
        
        Inputs:
            dx1, dx2, du - discrete step to discretize the space.
        
        """
#        self.state = np.array([0, 0])
        
        self.min_angle = -np.pi
        self.max_angle = np.pi
        self.max_torque = max_torque # in case of large control
        self.max_speed = max_speed # in case of angular velocity
        self.dt = dt
        
        self.dx1 = dx1
        self.dx2 = dx2
        self.du = du # control any computation in discrete space
        self.x1_decimals = len(str(dx1)) - 2
        self.x2_decimals = len(str(dx2)) - 2 
        self.u_decimals = len(str(du)) - 2
        
        self.DiscretizeSpace(dx1, dx2, du)
        
        
        # parameters of the inverted pendulum
        self.a = 4
        self.b = 1
        self.sigma = 1e-3 * np.eye(2)
        self.k = 3
        self.r = 1e-3
        
    def KeepInSpace(self, x, space='x'):
        if space == 'x':
            idx = np.sum(abs(x - self.state_space_array), axis=1).argmin()
            x = self.state_space_array[idx]
        elif space == 'u':
            idx = np.abs((x - self.control_space)).argmin()
            x = self.control_space[idx]
        
        return x, idx
        
    
    def DiscretizeSpace(self, dx1, dx2, du):
        """
        Discretize state space and control space
        """
        self.x1_range = np.round(np.arange(self.min_angle, self.max_angle+dx1, dx1), decimals=self.x1_decimals)
        self.x2_range = np.round(np.arange(-self.max_speed, self.max_speed+dx2, dx2), decimals=self.x2_decimals)
        xx1, xx2 = np.meshgrid(self.x1_range, self.x2_range)
        state_space = np.vstack((xx1.flatten(),xx2.flatten())).T
        state_space_list = []
        for i in range(state_space.shape[0]):
            x = state_space[i,:]
            x = (x[0], x[1])
            state_space_list.append(x)
        
        self.state_space = state_space_list
        self.state_space_array = state_space

        control_space = np.round(np.arange(-self.max_torque, self.max_torque+du, du), decimals=self.u_decimals)
        self.control_space = list(control_space)
        
        self.n1 = self.x1_range.shape[0]
        self.n2 = self.x2_range.shape[0]
        self.nu = control_space.shape[0]
        print("Size of angular space: ", self.n1)
        print("Size of speed space: ", self.n2)
        print("Size of control space:", self.nu)
        print("Discretization ratio:", 2*self.max_speed/self.n2*self.dt/(2*np.pi/self.n1))
        
        
    
    def reset(self):
        high = np.array([np.pi, 0])
        self.state = self.KeepInSpace(np.random.uniform(low=-high, high=high))
#        obs = self._get_obs()
#        return self.state, obs
        
    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])
    
    def step(self, x, u, tau=0.05):
        """
        Inputs:
            x - current state
            u - control input
            tau - time step.
        Outputs:
            new_state
            cost
        """
        
        theta = x[0]
        thetadot = x[1]
        
        u = np.clip(u, -self.max_torque, self.max_torque) # limit u in the space
        
        x1 = thetadot
        x2 = self.a * np.sin(theta) - self.b * thetadot + u
        # compute x+f(x,u)dt
        new_theta = x1 * self.dt + theta 
        new_thetadot = x2 * self.dt + thetadot        
        
        new_thetadot = np.clip(new_thetadot, -self.max_speed, self.max_speed)
                
        mean = np.array([new_theta, new_thetadot])
        

        
        # add some noise
        Sigma = self.sigma @ self.sigma.T * tau
        
        new_state = np.random.multivariate_normal(mean, Sigma).reshape(1,2)
              
        # Computation processing to keep the dynamic stable
        new_state[0,0] = np.round(self.angle_normalization(new_state[0,0]), decimals=self.x1_decimals)
        new_state[0,1] = np.clip(new_state[0,1], -self.max_speed, self.max_speed ) # limit u in the space
        new_state[0,1] = np.round(new_state[0,1], decimals=self.x2_decimals) # control any computation in discrete space
        new_state,_ = self.KeepInSpace(new_state.reshape(2,))
        self.state = new_state
        
        
        
        cost = (1 - np.exp(self.k * np.cos(theta) - self.k) + self.r/2 * u**2) * tau
        return self.state, cost

    
    def cal_gaussian(self, x, u, tau=0.05):

        theta = x[0]
        thetadot = x[1]
        
        u = np.clip(u, -self.max_torque, self.max_torque) # limit u in the space
#        u = np.round(u, decimals=self.u_decimals) # control any computation in discrete space
        
        x1 = thetadot
        x2 = self.a * np.sin(theta) - self.b * thetadot + u
#        # compute x+f(x,u)
        new_theta = x1 * self.dt + theta 
        new_thetadot = x2 * self.dt + thetadot 
        
#        new_thetadot = thetadot + (self.a * np.sin(theta) - self.b *thetadot + u) * tau
#        new_theta = theta + new_thetadot * tau
        
        mean = np.array([new_theta, new_thetadot])
        mean[0] = self.angle_normalization(mean[0])
        cov = self.sigma @ self.sigma.T * tau
        
        return self.KeepInSpace(mean)[0], cov
    
    def angle_normalization(self, x):
        """
        Cast angle that is not in [-pi, pi] to this range.
        """
        return (((x+np.pi) % (2*np.pi)) - np.pi)

class Planner():
    
    def __init__(self, env, gamma=1.0, tau=0.05):
        """
        Input:
            env - enviroment
            gamma - discount factor
        """
        self.gamma = gamma        
        self.env = env
        self.policy = {}
        self.tau = 0.05
#        self.gen_transition(env)
    
    def __call__(self, alg='VI'):
        self.policy_iter(alg)
    
    def gen_transition(self, env):
        nu = len(env.control_space)
        nx = len(env.state_space)
        P = np.zeros((nx, nx, nu))    
        
        stage_cost = np.zeros((nx, 1, nu))
        for idx_x in range(nx):
            x = env.state_space_array[idx_x]
            print(x)
            for idx_u in range(nu):
                u = env.control_space[idx_u]
                mean, cov = env.cal_gaussian(x, u, self.tau)
                p_x_prime = multivariate_normal.pdf(x, mean=mean, cov=cov)
                p_x_prime = p_x_prime/np.sum(p_x_prime)
                P[idx_x,:, idx_u] = p_x_prime
                cost = self.tau * (1 - np.exp(env.k*x[0]-env.k)+env.r/2*u**2)
                stage_cost[idx_x, 0, idx_u] = np.dot(p_x_prime, cost)
                
        
        self.P = P
        self.stage_cost = stage_cost
        
    
    def init_V(self):
        n = len(self.env.state_space)
#        self.V = np.zeros((n))
        self.V = np.random.randn(n)
    
    def policy_iter(self, alg='VI'):
        self.init_V()
        self.v_record = []
        self.v_r = [[],[]]
        print("V initialized.")
        if alg == 'VI':
            print("VI")
            self.VI()
        else:
            print("PI")
            self.PI()
    
    def VI(self, max_iter=200):
        self.init_V()
        self.policy = {}
        cts = 0
        map_cts = [1, 3, 5, 7]
        idx1 = self.env.state_space.index((0,0))
        idx2 = self.env.state_space.index((2,-1))
        while True:
            cts += 1
            if cts in map_cts:
                self.v_record.append( copy.deepcopy(self.V))
            e = self.value_update()
            self.v_r[0].append(copy.deepcopy(self.V[idx1]))
            self.v_r[1].append(copy.deepcopy(self.V[idx2]))
            print("Error:", e)

            
            if e < 1e-3 or cts >= max_iter:
                self.v_record.append( copy.deepcopy(self.V))
                break
            
        for x in self.env.state_space: # need to be modified
            self.FindOptPolicy(x)
    
    def value_update(self):
        v_past = copy.deepcopy(self.V)
        for idx_x in range(len(self.env.state_space)):
            x = self.env.state_space_array[idx_x,:]
#            print(x)
            v_min = np.inf
            for u in self.env.control_space:
#                print(u)
                mean, cov = self.env.cal_gaussian(x, u, tau=self.tau)
                v_tmp = self.expect(x, u, v_past, mean, cov)
                if v_tmp < v_min:
                    v_min = v_tmp
            
            self.V[idx_x] = v_min
        
        e = np.linalg.norm(self.V - v_past, np.inf)
        return e
    
    def expect(self, x, u, v, mean, cov):
        pf = multivariate_normal.pdf(self.env.state_space_array, mean, cov) # pf(x'|x,u)
        pf = pf/np.sum(pf)
        cost = self.tau * (1 - np.exp(self.env.k * np.cos(self.env.state_space_array[:,0])-self.env.k) + self.env.r/2 * u**2)
        return self.gamma * np.dot(pf, v) + np.dot(pf, cost)
    
    def FindOptPolicy(self, x):
        v_min = np.inf
        x_arr = np.array([x[0],x[1]])
        for u in self.env.control_space:
            mean, cov = self.env.cal_gaussian(x_arr, u, tau=self.tau)
            v_tmp = self.expect(x_arr, u, self.V, mean, cov)
            if v_tmp < v_min:
                v_min = v_tmp
                u_best = u
        
        self.policy[x] = u_best
    
    def PI(self, max_iter=15):
        self.init_V()
        self.policy = {}
        n = len(self.env.control_space)
        for x in self.env.state_space:
            self.policy[x] = self.env.control_space[np.random.choice(n)]
        
        cts = 0
        map_cts = [1, 3, 5, 7]
        idx1 = self.env.state_space.index((0,0)) #pick two states
        idx2 = self.env.state_space.index((2,-1))
        while True:
            cts +=1
            if cts in map_cts:
                self.v_record.append(copy.deepcopy(self.V))
            self.policy_eval()
            self.policy_improve()
            self.v_r[0].append(copy.deepcopy(self.V[idx1]))
            self.v_r[1].append(copy.deepcopy(self.V[idx2]))

            
            if self.flag or cts >= max_iter:
                self.v_record.append(copy.deepcopy(self.V))
                break

    def policy_eval(self, max_iter=20):
        cts = 0
        while True:
            v_past = copy.deepcopy(self.V)
            
            for idx_x in range(len(self.env.state_space)):
                x = self.env.state_space_array[idx_x,:]
                mean, cov = self.env.cal_gaussian(x, self.policy[self.env.state_space[idx_x]], tau=self.tau)
                self.V[idx_x] = self.expect(x, self.policy[self.env.state_space[idx_x]], self.V, mean, cov)
#                x_prime, cost = self.env.step(x, self.policy[x])
#                self.V[x] = cost + self.gamma * v_past[x_prime]
            
            e = np.linalg.norm(self.V - v_past, np.inf)
            print(e)
            cts += 1
            if e < 1e-3 or cts >= max_iter:
                break
       
    def policy_improve(self):
        self.flag = True
        for x in self.env.state_space:
            p_tmp = self.policy[x]
            self.FindOptPolicy(x)
            if p_tmp != self.policy[x]:
                self.flag *= False
    
    def policy_interp2d(self ,dx1, dx2, max_speed, plot=True):
#        policy_array = np.zeros((self.env.n1, self.env.n2))
#        for j in range(self.env.x2_range.shape[0]):
#            x2 = self.env.x2_range[j]
#            for i in range(self.env.x1_range.shape[0]):
#                x1 = self.env.x1_range[i]
#                policy_array[i, j] = self.policy[(x1, x2)]
        
        
        policy_array = np.zeros((self.env.n2, self.env.n1))
        for j in range(self.env.x2_range.shape[0]):
            x2 = self.env.x2_range[j]
            for i in range(self.env.x1_range.shape[0]):
                x1 = self.env.x1_range[i]
                policy_array[j, i] = self.policy[(x1, x2)]
        
        
        x1_decimals = len(str(dx1)) - 2
        x2_decimals = len(str(dx2)) - 2 
        
        f = interpolate.interp2d(self.env.x1_range, self.env.x2_range, policy_array, kind='cubic')  
        x1_new = np.round(np.arange(-np.pi, np.pi+self.env.dx1, dx1), decimals=x1_decimals)
        x2_new = np.round(np.arange(-max_speed, max_speed+self.env.dx2, dx2), decimals=x2_decimals)    
        xx, yy = np.meshgrid(x1_new,x2_new)
#        state_space = np.vstack((xx.flatten(),yy.flatten())).T
        policy_interp_2d = f(x1_new,x2_new)
        policy_interp = {}
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                x = (xx[i,j],yy[i,j])
                policy_interp[x] = policy_interp_2d[i,j]
        
        if plot:
            self.policy_2d = policy_interp_2d
            self.xx = xx
            self.yy = yy
            
#            fig = plt.figure()
#            ax = fig.gca(projection='3d')
#            ax.plot_surface(xx, yy, policy_interp_2d)
#            ax.set_xlabel('x1')
#            ax.set_ylabel('x2')
#            ax.set_zlabel('u')
#            plt.colorbar
#            ax.view_init(50, -60)
#            plt.show()
#            plt.tight_layout()
#            plt.savefig("Policy.png", transparent = True, bbox_inches = 'tight')
            
            
            level = np.arange(-self.env.max_torque, self.env.max_torque + 0.5, 0.5).tolist()
            fig, ax = plt.subplots()
            cs = ax.contourf(xx, yy, policy_interp_2d, level=level, cmap=plt.cm.bone)
            fig.colorbar(cs, ax=ax, shrink=0.9)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_title('Optimal Policy')
            plt.show()
            plt.tight_layout()
            plt.savefig("contour.png", transparent = True, bbox_inches = 'tight')
        
        
        return policy_interp
    
    def plot_V(self, v, idx):
#        v_array = np.zeros((self.env.n1, self.env.n2))
#        for j in range(self.env.x2_range.shape[0]):
#            for i in range(self.env.x1_range.shape[0]):
#                v_array[i, j] = v[i, j]
                
#        v_array = v.reshape(self.env.n1, self.env.n2, order='F').T
        v_array = v.reshape(self.env.n2, self.env.n1)
        
        dx1 = self.env.dx1
        dx2 = self.env.dx2
        x1_range = np.arange(-np.pi, np.pi + dx1, dx1)
        x2_range = np.arange(-self.env.max_speed, self.env.max_speed+dx2, dx2)
        xx, yy = np.meshgrid(x1_range, x2_range)
        
        fig, ax = plt.subplots()
        cs = ax.contourf(xx, yy, v_array, cmap=plt.cm.bone)
        fig.colorbar(cs, ax=ax, shrink=0.9)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
#        ax.set_zlabel('V')   
        plt.colorbar
        plt.show()
        plt.tight_layout()
        filename = "V_"+str(idx)+".png"
        plt.savefig(filename, transparent = True, bbox_inches = 'tight')
             

if __name__ == '__main__':

    dx1 = 0.1
    dx2 = 0.2
    du = 0.25
    max_speed = 4
    tau = 0.5
    env = IP(dx1, dx2, du, max_speed=max_speed, max_torque=3)
    env.reset()
    
    gamma = 0.9
    planner = Planner(env, gamma=gamma, tau=tau)
    alg = 'VI'
    planner(alg=alg)
    policy = planner.policy
    
    # interpolate and plot
    dx1_new = 0.01
    dx2_new = 0.1
    policy = planner.policy_interp2d(dx1_new, dx2_new, max_speed, plot=False)
#    
    for cts, v in enumerate(planner.v_record):
        planner.plot_V(v, cts)
    
    # simulation
    env = IP(dx1_new, dx2_new, du=0.001, max_speed=max_speed)
    env.reset()
    c_state = env.state[0]
    tau = 0.50
    t_range = np.arange(0, 10+tau, tau)
    theta = np.zeros((len(t_range)))
    u = np.zeros((len(t_range)))
    
    for i in range(len(t_range)):
        action = policy[tuple(c_state)]
        u[i] = action
        n_state,_ = env.step(c_state, action, tau=tau)
        theta[i] = c_state[0]
        c_state = n_state
    

    animation = EnvAnimate()
    animation.new_data(theta, u) 
    animation.start()
    