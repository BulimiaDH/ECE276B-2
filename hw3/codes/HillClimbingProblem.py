import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D

class Planner:
    
    def __init__(self, env, d_p=0.01, d_v=0.001, gamma=1.0, alpha=0.5, epsilon=0.5, num_episode=500):
        '''
        Initialization of all necessary variables to generate a policy:
        discretized state space X = S x V
        control space = {0(push left), 1(no push), 2(push right)}
        discount factor - gamma
        learning rate - alpha
        greedy probability (if applicable)
        Inputs:
            discrete - the single unit to discrete the state space. Default as 0.05
            gamma - discount factor. Default as 1.0
            alpha - step size or learning rate. 
        '''
        self.state_space = self.init_state(d_p, d_v)
        self.d_p = d_p
        self.d_v = d_v
        self.control_space = [0, 1, 2]
        self.env = env
        self.env._max_episode_steps = 20000
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_episode = num_episode
        self.Q = {} # Q(x,u).
        self.policy = {} # to store current best policy.
        
    def __call__(self, mc=False, on=True):
        """
        rn and return a policy via model-free policy iteration.
        """
        print("Called! Implementing policy iteration...")
        return self._td_policy_iter(on)
        
    def init_state(self, d_p, d_v):
        """
        Initialize state.
        """
        
        self.p_decimals = len(str(d_p)) - 2
        self.v_decimals = len(str(d_v)) - 2 # control decimals in case of floating problem
        
        prange = np.round(np.arange(-1.2, 0.6 + d_p, d_p), decimals=self.p_decimals)
        vrange = np.round(np.arange(-0.07, 0.07 + d_v, d_v), decimals=self.v_decimals)
        pp, vv = np.meshgrid(prange,vrange)
        state_space = np.vstack((pp.flatten(),vv.flatten())).T # (2, N), N is # states
        state_space_list = []
        for i in range(state_space.shape[0]):
            x = state_space[i,:]
            x = (x[0], x[1])
            state_space_list.append(x)
        
        return state_space_list
    
    def init_Q(self):
        """
        Initialize Q(x,u).
        """
        for x in self.state_space:
            self.Q[x] = {}
            for u in [0, 1, 2]:
                if x[0] == 0.5:
                    self.Q[x][u] = 0
                else:
                    self.Q[x][u] = np.random.randn()
#                    self.Q[x][u] = 0
        
#        self.Q = np.random.randn(self.state_space.shape[0],1,3) # randomly generate Q, vectorize it
    
    def _td_policy_iter(self, on=True):
        """
        TO BE IMPLEMENT
        Policy Iteration
        args: 
            on : on- vs. off-policy learning
        Returns: policy that minimizes Q wrt to controls
        """
        if on:
            self.SARSA()
            self.GetOptPolicy()
        else:
            self.Q_Learning()
            self.GetOptPolicy()
        
        print("Got Policy!")
        return self.policy

#    def gen_episode(self, policy):
#        pass
    
    def GetOptPolicy(self):
        """
        Given Q find the best policy
        """
        for x in self.state_space:
            self.policy[x] = self.FindOptAction(x)
                    
    def FindOptAction(self, x, flag=True):
        """
        Given state x, return best action.
        """
        if x[0] >= 0.50:
            best_u = 2
            Q_min = self.Q[x][best_u]
        else:
            Q_min = np.inf
            for u in [0, 1, 2]:
                if self.Q[x][u] < Q_min:
                    Q_min = self.Q[x][u]
                    best_u = u
        if flag == True:
            return int(best_u)
        else:
            return best_u, Q_min
    
    def SARSA(self):
        self.init_Q() # initialize Q
        epis_num = self.num_episode
        plot_time = [0, 50, 100, 2000, 5000, 10000]
        epsi = 0
        self.q_record = [copy.deepcopy(self.Q)]
        self.iter_epis = []
        
        q = copy.deepcopy(self.Q)
        q1 = q[(0.0,0.0)]
        q2 = q[(-1.0,0.05)]
        q3 = q[(0.30,-0.05)]
        q_record_list = [(0.0,0.0),(-1.0,0.05),(0.30,-0.05)]
        self.q_record_s = [[[q1[0]],[q1[1]],[q1[2]]],[[q2[0]],[q2[1]],[q2[2]]],[[q3[0]],[q3[1]],[q3[2]]]]
        
#        np.random.seed(0)
        while True: # iterate episode
            epsi += 1
            x = self.env.reset() # choose a state x 
            x[0] = np.round(x[0], decimals=self.p_decimals)
            x[1] = np.round(x[1], decimals=self.v_decimals) # keep its decimals in discrete state space
            x = (x[0], x[1])
            self.env.state = x
            u = self.eps_greedy(x, eps=self.epsilon) # choose a action based on e-greedy
#            done = False
#            while done== False: # until ternimal
            cts = 0
            while x[0] < 0.50:
#            while cts <= 5000 and x[0] < 0.50:
                cts += 1
                x_prime, reward, _, _=self.env.step(u)
                x_prime[0] = np.round(x_prime[0], decimals=self.p_decimals)
                x_prime[1] = np.round(x_prime[1], decimals=self.v_decimals)
                x_prime = (x_prime[0], x_prime[1])
#                self.env.state = x_prime
                u_prime = self.eps_greedy(x_prime, eps=self.epsilon)
                self.Q[x][u] = self.Q[x][u] + self.alpha * (-reward + self.gamma * self.Q[x_prime][u_prime] - self.Q[x][u])
                x = x_prime
                u = u_prime
            
            self.iter_epis.append(cts)
            print("One Episode Finished! #Iteration:", cts)
            print('Episode:', epsi)
            
            q = copy.deepcopy(self.Q)
            for idx, state in enumerate(q_record_list):
                self.q_record_s[idx][0].append(q[state][0])
                self.q_record_s[idx][1].append(q[state][1])
                self.q_record_s[idx][2].append(q[state][2])
                
#            if epsi % plot_time == 0:
            if epsi in plot_time:
                self.q_record.append(copy.deepcopy(self.Q))
            if epsi >= epis_num:
                break
            

    
    def eps_greedy(self, x, eps=0.3):
        """
        ε-greedy.
        Input:
            x - state
            eps - epsilon
        """
        thresh = 1 - eps + eps/3
        if x[0] >= 0.50:
            u = 2
        else:
            best_u = self.FindOptAction(x)
            if np.random.uniform(0, 1) > thresh:
                u_list = [0, 1, 2]
                u_list.remove(best_u)
                u = random.sample(u_list, 1)[0]
            else:
                u = best_u
        
        return int(u)
    
    def Q_Learning(self):
        """
        Implementation of Q Learning.
        """
        self.init_Q()
        epis_num = self.num_episode
        plot_time = epis_num/5
        epsi = 0
        self.q_record = [copy.deepcopy(self.Q)]
        
        q = copy.deepcopy(self.Q)
        q1 = q[(0.0,0.0)]
        q2 = q[(-1.0,0.05)]
        q3 = q[(0.30,-0.05)]
        q_record_list = [(0.0,0.0),(-1.0,0.05),(0.30,-0.05)]
        self.q_record_s = [[[q1[0]],[q1[1]],[q1[2]]],[[q2[0]],[q2[1]],[q2[2]]],[[q3[0]],[q3[1]],[q3[2]]]]
        
        self.iter_epis = []
        while True:
            epsi += 1
            x = self.env.reset() # choose a state x 
            x[0] = np.round(x[0], decimals=self.p_decimals)
            x[1] = np.round(x[1], decimals=self.v_decimals) # keep its decimals in discrete state space
            x = (x[0], x[1])
            self.env.state = x
#            done = False
                       
            cts = 0
#            while done== False: # until ternimal
            while x[0] < 0.50:
                cts += 1
                u = self.eps_greedy(x, eps=self.epsilon) # choose a action based on e-greedy
                x_prime, reward, _, _=self.env.step(u)
                x_prime[0] = np.round(x_prime[0], decimals=self.p_decimals)
                x_prime[1] = np.round(x_prime[1], decimals=self.v_decimals)
                x_prime = (x_prime[0], x_prime[1])
                self.env.state = x_prime
                _, Q_min = self.FindOptAction(x_prime, flag=False)
                self.Q[x][u] = self.Q[x][u] + self.alpha * (-reward + self.gamma * Q_min - self.Q[x][u])
                x = x_prime
            
            self.iter_epis.append(cts)
            print("One Episode Finished! #Iteration:", cts)
            print('Episode:', epsi)
            
            q = copy.deepcopy(self.Q)
            for idx, state in enumerate(q_record_list):
                self.q_record_s[idx][0].append(q[state][0])
                self.q_record_s[idx][1].append(q[state][1])
                self.q_record_s[idx][2].append(q[state][2])
            
            if epsi % plot_time == 0:
                self.q_record.append(copy.deepcopy(self.Q))
            if epsi >= epis_num:
                break

    def rollout(self, env, policy=None, render=False):
        """
        ple trajectory based on a policy
        """
        traj = []
        t = 0
        done = False
        c_state = env.reset()
        if policy is None:
            while not done and t < 200: 
                action = env.action_space.sample()
                if render:
                    env.render()
                n_state, reward, done, _ = env.step(action)
                traj.append((c_state, action, reward))
                c_state = n_state
                t += 1

            env.close()
            return traj

        else:
            c_state[0] = np.round(c_state[0], decimals=self.p_decimals)
            c_state = (c_state[0], c_state[1])
            env.state = c_state
            while not done and t < 300:
                action = policy[c_state]
                if render:
                    env.render()

                n_state, reward, done, _ = env.step(action)
                n_state[0] = np.round(n_state[0], decimals=self.p_decimals)
                n_state[1] = np.round(n_state[1], decimals=self.v_decimals)
                n_state = (n_state[0], n_state[1])
                traj.append((c_state, action, reward))
                c_state = n_state
                env.state = c_state
                t += 1

            env.close()
            
            traj.append((c_state, action, reward))
            return traj
    
    def plot_Q(self, Q):
        prange = np.round(np.arange(-1.2, 0.6 + self.d_p, self.d_p), decimals=self.p_decimals)
        vrange = np.round(np.arange(-0.07, 0.07 + self.d_v, self.d_v), decimals=self.v_decimals)
        pp, vv = np.meshgrid(prange,vrange)
        pv = np.stack((pp,vv),axis=2)
        q0 = np.zeros((pp.shape[0], pp.shape[1]))
        q1 = np.zeros((pp.shape[0], pp.shape[1]))
        q2 = np.zeros((pp.shape[0], pp.shape[1]))
        
        for i in range(pp.shape[0]):
            for j in range(pp.shape[1]):
                xy = (pv[i,j,0],pv[i,j,1])
                q0[i,j] = Q[xy][0]
                q1[i,j] = Q[xy][1]
                q2[i,j] = Q[xy][2]
        
        fig = plt.figure()
        ax = fig.add_subplot(311, projection='3d')
        ax.plot_surface(pp, vv, q0)
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('Q')
        ax.set_title('u = 0')
        
        ax = fig.add_subplot(312, projection='3d')
        ax.plot_surface(pp, vv, q1)
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('Q')
        ax.set_title('u = 1')
        
        ax = fig.add_subplot(313, projection='3d')
        ax.plot_surface(pp, vv, q2)
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('Q')
        ax.set_title('u = 2')        
        
    def plot_Q_curve(self):
        
        for i in range(3):
            fig = plt.figure()
            ax = fig.add_subplot(311)
            ax.plot(self.q_record_s[i][0])
            ax.set_xlabel('Episode')
            ax.set_ylabel('Q')
            ax.set_title('u = 0')
            plt.grid(True)
            
            ax = fig.add_subplot(312)
            ax.plot(self.q_record_s[i][1])
            ax.set_xlabel('Episode')
            ax.set_ylabel('Q')
            ax.set_title('u = 1')
            plt.grid(True)
            
            ax = fig.add_subplot(313)
            ax.plot(self.q_record_s[i][2])
            ax.set_xlabel('Episode')
            ax.set_ylabel('Q')
            ax.set_title('u = 2')
            plt.grid(True)
            plt.show()
            
            
            title = "../results/p3/state"+str(i)+"d_p"+str(d_p)+"_d_v"+str(d_v)+"_gamma"+str(gamma)+"_alpha"+str(alpha)+"_eps"+str(epsilon)+".png"
            plt.tight_layout()
            plt.savefig(title, transparent = True, bbox_inches = 'tight') 
        
    
    def plot_policy(self):
        """
        Plot the optimal policy distribution in state space.
        Red dots represent u = 0 (push left)
        Green dots represent u = 1 (no push)
        Blue dots represent u = 2 (push right)
        """
        
        prange = np.round(np.arange(-1.2, 0.6 + self.d_p, self.d_p), decimals=self.p_decimals)
        vrange = np.round(np.arange(-0.07, 0.07 + self.d_v, self.d_v), decimals=self.v_decimals)
        
        U, V = np.meshgrid(prange,vrange)
        vec = np.stack((U,V),axis=2)
        for i in range(vec.shape[0]):
            for j in range(vec.shape[1]):
                xy = (vec[i,j,0],vec[i,j,1])
                pi = self.policy[xy]
                if pi == 0:
                    U[i, j] = -1
                elif pi == 1:
                    U[i, j] = 0
                elif pi == 2:
                    U[i, j] = 1
        
        V = 0 * V
                
        fig, ax = plt.subplots(figsize=(20,10))
        ax.quiver(prange, vrange, U, V, alpha=1, scale=30, units='xy')
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        
        plt.show()
        title = "../results/p3/d_p_"+str(d_p)+"_d_v_"+str(d_v)+"_gamma_"+str(gamma)+"_alpha_"+str(alpha)+"_eps_"+str(epsilon)+".png"
        plt.tight_layout()
        plt.savefig(title, transparent = True, bbox_inches = 'tight') 

if __name__ == '__main__':

    env = gym.make('MountainCar-v0')

    np.random.seed(0)
    
    on = False # on-policy or off-policy
    # hyperparameters
    num_episode = 20000
    epsilon = 0.1 # epsilon-greedy
    d_p = 0.1 # discretize position space
    d_v = 0.01 # discretize velocity space
    gamma = 1.0 # discount factor
    alpha = 0.1 # learning rate
    planner = Planner(env, d_p=d_p, d_v=d_v, gamma=gamma, alpha=alpha, epsilon=epsilon, num_episode= num_episode)
    planner.__call__(on=on)
    policy = planner.policy
    
    traj_len = 0
    num_test = 100 # test 100 times
    for i in range(num_test):
        traj = planner.rollout(env, policy=policy, render=False)
        traj_len += len(traj)
        
    traj_len = traj_len/num_test # mean value
        
    with open('./p3_record.txt', 'a') as f:
        condition = "On="+str(on)+" d_p="+str(d_p)+" d_v="+str(d_v)+" gamma="+str(gamma)+" alpha="+str(alpha)+" eps="+str(epsilon)+"\n"
        f.write(condition)
        f.write("#Iterations in rullout:"+str(traj_len)+"\n")
    print(traj)
    
    # plot results
    planner.plot_policy() # plot policy
    planner.plot_Q_curve()
#    for q in planner.q_record:
#        planner.plot_Q(q)
        
    fig = plt.figure()
    plt.plot(planner.iter_epis)
    plt.xlabel('Episode')
    plt.ylabel('Iterations')
    plt.grid(True)
    plt.show()
