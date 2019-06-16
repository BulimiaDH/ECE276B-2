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

class EnvAnimate:

    '''
    Initialize Inverted Pendulum
    '''
    def __init__(self):

        # Change this to match your discretization
        self.dt = 0.05
        self.t = np.arange(0.0, 2.0, self.dt)

        # Random trajectory for example
        self.theta = np.linspace(-np.pi, np.pi, self.t.shape[0])
        self.x1 = np.sin(self.theta)
        self.y1 = np.cos(self.theta)
        self.u = np.zeros(self.t.shape[0])

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2, -2, 2])
        
        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.8, '', transform=self.ax.transAxes)


    def new_data(self, theta):
        '''
            Provide new rollout theta values to reanimate
        '''
        self.theta = theta # theta should be an array with shape(t.shape[0],)
        self.x1 = np.sin(theta)
        self.y1 = np.cos(theta) # this is what's gonna show in xy-axis.
        self.u = np.zeros(self.t.shape[0])
#        self.u = np.zeros(t.shape[0])

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
        # self.ani.save('double_pendulum.mp4', fps=15)
        plt.show()

class IP:
    """
    Environment for Inverted Pendulum.
    """
    def __init__(self, dx1, dx2, du):
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
        self.max_torque = 2 # in case of large control
        self.max_speed = 8 # in case of angular velocity
        self.dt = 0.05
        
        self.dx1 = dx1
        self.dx2 = dx2
        self.du = du # control any computation in discrete space
        self.x1_decimals = len(str(dx1)) - 2
        self.x2_decimals = len(str(dx2)) - 2 
        self.u_decimals = len(str(du)) - 2
        
        self.DiscretizeSpace(dx1, dx2, du)
        
        
        # parameters of the inverted pendulum
        self.a = 1
        self.b = 0.5
        self.sigma = 0.05 * np.eye(2)
        self.k = 1.0
        self.r = 0.1
        
    def KeepInSpace(self, x, space='x'):
        if space == 'x':
            idx = np.sum((x - self.state_space), axis=1).argmin()
            x = self.state_space[idx]
        elif space == 'u':
            idx = np.abs((x - self.control_space)).argmin()
            x = self.control_space[idx]
        
        return x, idx
        
    
    def DiscretizeSpace(self, dx1, dx2, du):
        """
        Discretize state space and control space
        """
        x1_range = np.round(np.arange(self.min_angle, self.max_angle+dx1, dx1), decimals=self.x1_decimals)
        x2_range = np.round(np.arange(-self.max_speed, self.max_speed+dx2, dx2), decimals=self.x2_decimals)
        xx1, xx2 = np.meshgrid(x1_range, x2_range)
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
        
        n1 = x1_range.shape[0]
        n2 = x2_range.shape[0]
        nu = control_space.shape[0]
        print("Size of angular space: ", n1)
        print("Size of speed space: ", n2)
        print("Size of control space:", nu)
        print("Discretization ratio:", 2*self.max_speed/n2*self.dt/(2*np.pi/n1))
        
#        self.state_space_dict = {}
#        for i in range(len(self.state_space_list)):
#            self.state_space_dict[self.state_space_list[i]] = i
#        self.control_space_dict = {}
        
    
    def reset(self):
        high = np.array([np.pi, 1])
        self.state = np.random.uniform(low=-high, high=high)
#        obs = self._get_obs()
#        return self.state, obs
        
    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])
    
    def step(self, x, u, tau=.05, gaussian=False):
        """
        Inputs:
            u - control input
            tau - time step.
        Outputs:
            new_state
            cost
        """
        
#        theta = self.state[0]
#        thetadot = self.state[1]
        theta = x[0]
        thetadot = x[1]
        
        u = np.clip(u, -self.max_torque, self.max_torque) # limit u in the space
        u = np.round(u, decimals=self.u_decimals) # control any computation in discrete space
        
        x1 = thetadot
        x2 = self.a * np.sin(theta) - self.b * thetadot + u
        
        # compute x+f(x,u)
        new_theta = (x1 + theta) * tau
        new_thetadot = (x2 + thetadot) * tau
        
        # add some noise
        Sigma = self.sigma @ self.sigma.T * tau
        
        if not gaussian:
            R = np.linalg.cholesky(Sigma)
            new_state = np.dot(np.random.randn(1, 2), R) + np.array([[new_theta, new_thetadot]])
            
            # Computation processing to keep the dynamic stable
            new_state[0,0] = np.round(self.angle_normalization(new_state[0,0]), decimals=self.x1_decimals)
            new_state[0,1] = np.clip(new_state[0,1], -self.max_speed, self.max_speed ) # limit u in the space
            new_state[0,1] = np.round(new_state[0,1], decimals=self.x2_decimals) # control any computation in discrete space
            new_state,_ = self.KeepInSpace(new_state.reshape(2,))
            self.state = new_state
            
            
            
            cost = (1 - np.exp(self.k * np.cos(theta) - self.k) + self.r/2 * u**2) * tau
            return self.state, cost
        else:
            mean = np.array([new_theta, new_thetadot])
            cov = Sigma
            cost = 1 - np.exp(self.k * np.cos(theta) - self.k) + self.r/2 * u**2
            return mean, cov, cost
    
#    def cal_gaussian(self, x, u, tau):
#        theta = x[0]
#        thetadot = x[1]
#        
#        u = np.clip(u, -self.max_torque, self.max_torque) # limit u in the space
#        u = np.round(u, decimals=self.u_decimals) # control any computation in discrete space
#        
#        x1 = thetadot
#        x2 = self.a * np.sin(theta) - self.b * thetadot + u
#        
#        # compute x+f(x,u)
#        new_theta = (x1 + theta) * tau
#        new_thetadot = (x2 + thetadot) * tau
#        
#        mean = np.array([new_theta, new_thetadot])
#        cov = self.sigma @ self.sigma.T * tau
#        
#        return mean, cov
    
    def angle_normalization(self, x):
        """
        Cast angle that is not in [-pi, pi] to this range.
        """
        return (((x+np.pi) % (2*np.pi)) - np.pi)

class Planner():
    
    def __init__(self, env, gamma=1.0):
        """
        Input:
            env - enviroment
            gamma - discount factor
        """
        self.gamma = gamma        
        self.env = env
        self.policy = {}
#        self.init_V()
        
#        self.env_estimate()
        
#    def env_estimate(self):
#        """
#        Try to estimate the transition matrix and cost vector by MLE.
#        """
#        self.reset()
#        nu = self.control_space.shape[0]
#        nx = self.state_space.shape[0]
#        P = np.zeros((nx, nx, nu))
#        
#        stage_cost = np.zeros((nx, 1, nu))
#        idx_x = 0
#        for x in self.state_space_list:
#            idx_u = 0
#            for u in self.control_space_list:
#                count = np.zeros((nx,))
#                for i in range(1000):
#                    x_prime, cost = self.step(x, u)
#                    _, idx_x_prime = self.KeepInSpace(x_prime)
#                    count[idx_x_prime] += 1
#                    
#                count = count/np.sum(count)
#                P[idx_x,:,idx_u] = count
#                stage_cost[idx_x, 0, idx_u] = cost
#                idx_u += 1
#            idx_x += 1
#            
#        self.trasition = P
#        self.cost = stage_cost
        
    
    def init_V(self):
#        n = len(self.env.state_space)
        self.V = {}
        for x in self.env.state_space:
            self.V[x] = 0
#            self.V[x] = np.random.randn()
    
    def policy_iter(self, flag=True):
        self.init_V()
        if flag:
            self.VI()
        else:
            self.PI()
    
    def VI(self, max_iter=10):
        self.init_V()
        self.policy = {}
        cts = 0
        while True:
            cts += 1
            e = self.value_update()
            print(e)
            if e < 1e-5 or cts >= max_iter:
                break
            
        for x in self.env.state_space:    
            self.FindOptPolicy(x)
    
    def value_update(self):
        v_past = copy.deepcopy(self.V)
#        e = 0
        for x in self.env.state_space:
            print(x)
            v_min = np.inf
            for u in self.env.control_space:
                print(u)
                
                x_prime, cost = self.env.step(x, u)
                v_tmp = cost + self.gamma * v_past[x_prime]
#                mean, cov, cost = self.env.step(x, u, gaussian=True)
                
                v_tmp = cost + self.gamma * self.expect(self.env.state_space_array, mean, cov)
                
                if v_tmp < v_min:
                    v_min = v_tmp
#                    e = max([e, abs(v_min - v_tmp)])
            
            self.V[x] = v_min
        e = self.error_cal(v_past)
        return e
    
    def expect(self, x, mean, cov):
        pf = multivariate_normal(x, mean, cov) # pf(x'|x,u)
        return pf.dot(self.V)
    
    def FindOptPolicy(self, x):
        v_min = np.inf
        for u in self.env.control_space:
            x_prime, cost = self.env.step(x, u)
            v_tmp = cost + self.gamma * self.V[x_prime]
            if v_tmp < v_min:
                v_min = v_tmp
                u_best = u
        
        self.policy[x] = u_best
    
    def PI(self, max_iter=100):
        self.init_V()
        self.policy = {}
        n = len(self.env.control_space)
        for x in self.env.state_space:
            self.policy[x] = self.env.control_space[np.random.choice(n)]
        
        cts = 0
        while True:
            cts +=1
            self.policy_eval()
            self.policy_improve()
            if self.flag or cts >= max_iter:
                break

    def policy_eval(self, max_iter=100):
        cts = 0
        while True:
            v_past = copy.deepcopy(self.V)
#            e = 0
            
            for x in self.env.state_space:
#                print(x)
#                v = self.V[x]
                x_prime, cost = self.env.step(x, self.policy[x])
                self.V[x] = cost + self.gamma * v_past[x_prime]
#                e = max([e, abs(v - self.V[x])])
            
            e = self.error_cal(v_past)
            print(e)
            cts += 1
            if e < 1e-5 or cts >= max_iter:
                break

    def error_cal(self, v):
        V_ = np.zeros((len(self.env.state_space)))
        v_ = np.zeros((len(self.env.state_space)))
        cts = 0
        for i in self.env.state_space:
            V_[cts] = self.V[i]
            v_[cts] = v[i]
            cts += 1
            
        e = np.linalg.norm(V_ - v_, np.inf)
        return e
    
    
    def policy_improve(self):
        self.flag = True
        for x in self.env.state_space:
            p_tmp = self.policy[x]
            self.policy[x] = self.FindOptPolicy(x)
            if p_tmp != self.policy[x]:
                self.flag = False


if __name__ == '__main__':
#	animation = EnvAnimate()
#	animation.start()
    dx1 = 0.02
    dx2 = 0.5
    du = 0.1
    env = IP(dx1, dx2, du)
    env.reset()
    
    gamma = 1.0
    planner = Planner(env, gamma=gamma)
    planner.VI()
#    planner.PI()
    policy = planner.policy
    
    # simulation
    env.reset()
    c_state = env.state
    t_range = np.arange(0, 2+0.05, 0.05)
    theta = np.zeros((len(t_range)))
    for i in range(len(t_range)):
        action = policy[c_state]
        n_state,_ = env.step(action)
        theta[i] = n_state
        n_state = c_state
    
    np.save('theta_vi.npy', theta)