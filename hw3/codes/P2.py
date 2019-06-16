# -*- coding: utf-8 -*-
"""
Created on Fri May 31 21:06:51 2019

@author: epyir
"""
import numpy as np
import copy
import matplotlib.pyplot as plt

class env:
    """
    Environment for problem 2.
    """
    def __init__(self):
        self.states = [i for i in range(1,26)]
        self.A = 2 # special state A
        self.A_prime = 22
        self.B = 4 # special state B
        self.B_prime = 14
        
        self.n_bound = [1,2,3,4,5] # boundaries
        self.s_bound = [21,22,23,24,25]
        self.w_bound = [1,6,11,16,21]
        self.e_bound = [5,10,15,20,25]
        
        # Assign an agent as robot
#        self.x = np.random.randint(1, 26) # random initial state
#        self.cost = 0
        
    def motion_model(self, x, u):
        """
        Motion model, given x and control input u.
        Return next position and cost at this step.
        """
        cost = 0
        if x is not self.A and x is not self.B: # if x is not special states
            if u == 'n':
                if x not in self.n_bound:
                    x = x - 5
                else:
                    cost = 1
                    
            if u == 's':
                if x not in self.s_bound:
                    x = x + 5
                else:
                    cost = 1
            
            if u == 'e':
                if x not in self.e_bound:
                    x = x + 1
                else:
                    cost = 1
                   
            if u == 'w':
                if x not in self.w_bound:
                    x = x - 1
                else:
                    cost = 1                
            
        elif x == self.A:
            x = self.A_prime
            cost = -10

        elif x == self.B:
            x = self.B_prime
            cost = -5

        
#        self.x = x
#        self.cost += cost
        return x, cost
    
    def show_policy(self, title):
        fig, ax = plt.subplots()
        ax.set_xlim(0,5)
        ax.set_ylim(0,5)
        ax.grid()
        xrange = [0.5, 1.5, 2.5, 3.5, 4.5]
        yrange = [4.5, 3.5, 2.5, 1.5, 0.5]
        cts = 0
        for y_cts in range(5):
            for x_cts in range(5):
                s = self.states[cts]
                u = self.U[str(s)]
                if u == 'n':
                    start = np.array([xrange[x_cts], yrange[y_cts]-0.5])
                    end = np.array([xrange[x_cts],yrange[y_cts]+0.5])
                if u == 's':
                    start = np.array([xrange[x_cts],yrange[y_cts]+0.5])
                    end = np.array([xrange[x_cts], yrange[y_cts]-0.5])
                if u == 'w':
                    start = np.array([xrange[x_cts]+0.5,yrange[y_cts]])
                    end = np.array([xrange[x_cts]-0.5,yrange[y_cts]])
                if u == 'e':
                    start = np.array([xrange[x_cts]-0.5,yrange[y_cts]])
                    end = np.array([xrange[x_cts]+0.5,yrange[y_cts]])
                
                ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                         length_includes_head=True,
                         head_width=0.25, head_length=0.25, fc='r', ec='b')
                
                cts += 1
        plt.show()            
        plt.tight_layout()
        plt.savefig(title, transparent = True, bbox_inches = 'tight') 
    
class VI(env):
    def __init__(self, gamma=0.9, thresh=1e-12):
        env.__init__(self)
        self.thresh = thresh
        self.gamma = gamma
        self.V = {}
        for i in range(1,26):
#            self.V[str(i)] = 0
            self.V[str(i)] = np.random.randn(1)[0]
        
        self.U = {}
    
    def value_update(self):

        v = copy.deepcopy(self.V)
        for x in self.states: # for all states
            v_min = np.inf
            for u in ['n','s','e','w']: # for every control
                x_prime, cost = self.motion_model(x, u) # all possible x'
                v_tmp = cost + self.gamma * v[str(x_prime)] # compute value-function
                if v_tmp < v_min:
                    v_min = v_tmp
                    
            self.V[str(x)] = v_min
        
        e = self.error_cal(v)
        return e
    
    def error_cal(self, v):
        V_ = np.zeros((25))
        v_ = np.zeros((25))
        for i in range(25):
            V_[i] = self.V[str(i+1)]
            v_[i] = v[str(i+1)]
            
        e = np.linalg.norm(V_ - v_, np.inf)
        return e
    
    def VI_iter(self, max_iter=10000):
        cts = 0
        # update value function
        while True:
            cts += 1
            e = self.value_update()
            print(e)
            if e < self.thresh or cts >= max_iter:
#            if cts >= max_iter:
                print('Total iter:', cts)
                break
        
        for x in self.states:
            v_min = np.inf
            for u in ['s','n','e','w']: # for every control
                x_prime, cost = self.motion_model(x, u)
                v_tmp = cost + self.gamma * self.V[str(x_prime)] # compute value-function
                if v_tmp < v_min:
                    v_min = v_tmp
                    u_best = u
            
            self.U[str(x)] = u_best
            


        
class PI(env):
    def __init__(self, gamma=0.9, thresh=1e-12):
        env.__init__(self)
        self.thresh = thresh
        self.U = {} 
        self.gamma = gamma
        self.V = {}
        for i in range(1,26): # initialization
#            self.V[str(i)] = 0
            self.V[str(i)] = np.random.randn(1)[0]
            self.U[str(i)] = np.random.permutation(['w','n','s','e'])[0] # random policy
    
    def policy_eval(self, policy, max_iter=10000):
        """
        Policy evaluation, given policy.
        """
        cts = 0
        while True:
            self.v_past = copy.deepcopy(self.V)
            e = 0
            
            for x in self.states: # for each state
                v = self.V[str(x)]
                self.V[str(x)] = self.value_update(x, self.U)
                e = max([e, abs(v - self.V[str(x)])])
            
#            print('iter:', cts)
            print('error:', e)
#            print('values:', self.V)
            cts += 1
            if e < self.thresh or cts >= max_iter:
                break
    
    def value_update(self, x, policy):
        """
        Update value function given policy.
        """
        
        x_prime, cost = self.motion_model(x, policy[str(x)])
        v = cost + self.gamma * self.v_past[str(x_prime)]
        
        return v
    
    def policy_improve(self):
        """
        Policy improvement.
        """
        self.flag = True
        for x in self.states:
            p_tmp = self.U[str(x)]
            self.U[str(x)] = self.opt_policy(x)
            if p_tmp != self.U[str(x)]:
                self.flag = False
                break
    
    def opt_policy(self, x):
        """
        Search for currently best policy.
        """
        v = np.inf
        for u in ['s','n','w','e']:
            v_tmp = self.value_update(x, {str(x): u})
            if v_tmp < v:
                v = v_tmp
                best_policy = u
        
#        print(best_policy)
        return best_policy
        
    
    def PI_iter(self, max_iter=10000):
        cts = 0
        while True:
            cts += 1
            print(cts)
            self.policy_eval(self.U)
            self.policy_improve()
            print("policy is", self.U)
            if self.flag or cts >= max_iter:
                print("Total iter:", cts)
                break
        
class QI(env):
    def __init__(self, gamma=0.9, thresh=1e-12):
        env.__init__(self)
        self.gamma = gamma
        self.thresh = thresh
        self.U = {} # Current best policy for each state
        self.Q = {}
        for x in self.states: # Initialize Q-value
            self.Q[str(x)] = {}
            for u in ['n', 's', 'w', 'e']:
                self.Q[str(x)][u] = np.random.randn()
#                self.Q[str(x)][u] = 0
                
    def QI_iter(self, max_iter=10000):
        self.q_eval(max_iter)
        self.policy_improve()
        
    def Q2V(self):
        V = {}
        for x in self.states:
            v_min = np.inf
            for u in ['n', 's', 'w', 'e']:
                if self.Q[str(x)][u] < v_min:
                    v_min = self.Q[str(x)][u]
            
            V[str(x)] = v_min
        
        return V
        
            
    def q_eval(self, max_iter=10000):
        """
        Q-value evaluation.
        """
        cts = 0
        while True:
            self.q_past = copy.deepcopy(self.Q)
            e = 0
            cts += 1
            for x in self.states:
                for u in ['n', 's', 'w', 'e']: # current state action u
                    q_min = np.inf
                    x_prime, cost = self.motion_model(x, u)
                    for pi in ['n', 's', 'w', 'e']: # next state policy pi
                        # find the policy pi that leads to minimal Q(x',pi)
                        q_tmp = self.q_past[str(x_prime)][pi]
                        if q_tmp < q_min:
                            q_min = q_tmp
                    
                    # Update current state's Q-value
                    self.Q[str(x)][u] = cost + self.gamma * q_min
                    e = max([e, abs(self.Q[str(x)][u] - self.q_past[str(x)][u])])
                    
            print("iter:", cts)
#            print("error:", e)
            if e < self.thresh or cts >= max_iter:
                break
    
    def policy_improve(self):
        for x in self.states:
            q_min = np.inf
            for u in ['s','n','e','w']: # for every action
                q_tmp = self.Q[str(x)][u]
                if q_tmp < q_min:
                    q_min = q_tmp
                    best_policy = u
            
            self.U[str(x)] = best_policy
        
def VI_test(gamma=0.9):
    vi_cal = VI(gamma)
    vi_cal.VI_iter()
    opt_value = vi_cal.V
    opt_policy = vi_cal.U
    print(opt_value)
    print(opt_policy)
    vi_cal.show_policy(title='../results/p2/gamma_'+str(gamma)+'_VI.png')

def PI_test(gamma=0.9):
    pi_cal = PI(gamma)
    pi_cal.PI_iter()
    opt_value = pi_cal.V
    opt_policy = pi_cal.U
    print(opt_value)
    print(opt_policy)    
    pi_cal.show_policy(title='../results/p2/gamma_'+str(gamma)+'_PI.png')
    
def QI_test(gamma=0.9):
    qi_cal = QI(gamma)
    qi_cal.QI_iter()
#    opt_value = qi_cal.Q
    opt_V_value = qi_cal.Q2V()
    opt_policy = qi_cal.U
#    print(opt_value)
    print(opt_V_value)
    print(opt_policy)
    qi_cal.show_policy(title='../results/p2/gamma_'+str(gamma)+'_QI.png')

    
if __name__ == '__main__':
    gamma = 0.9
#    VI_test(gamma=gamma)
    PI_test(gamma=gamma)
#    QI_test(gamma=gamma)