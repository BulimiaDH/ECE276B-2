# -*- coding: utf-8 -*-
"""
This is the code for Problem 1.
"""
import numpy as np
import copy

class PI:
    """
        Policy Iteration algorithm.
    """
    def __init__(self, states, thresh=1e-12, gamma=1.0):
        """
        Input:
            states - a list of strings, each number string presents one state
            trasition - trasition matrix
            thresh - threshold for Policy Evaluation.
        """
        self.states = states
        self.policy = {}
        self.V, self.U = self.initialization_VU(states)
        self.P, self.l = self.initialization_transition()
        self.thresh = thresh
        self.gamma = gamma

    def initialization_VU(self, states):
        """
        Initialize Value V(x) and Control space U(x).
        """
        V = {}
        U = {}
        for x in states:
#            V[x] = np.random.randn()
#            V[x] = int(x)
            lst = []
            if int(x) > 0 and int(x) < 3:
                V[x] = 0
                for m in range(1, int(x) + 1):
                    lst.append((0, m))
                    lst.append((1, m))
                    
                self.policy[x] = (1, 1)
                
            else:
                V[x] = -int(x)
            U[x] = lst
            
        return V, U
    
    def initialization_transition(self):
        P = {}
        l = {}
        for x in self.states:
            if int(x)>0 and int(x)<3:
                P[x] = {}
                l[x] = {}
                for pi in self.U[x]: # choose one policy
                    P[x][pi] = {}
                    l[x][pi] = {}
#                    for m in range(1, pi[1]+1): # bet $m
                    m = pi[1]
                    if pi[0] == 1: # bet on red
                        P[x][pi][str(int(x)+m)] = 0.7 # win
                        P[x][pi][str(int(x)-m)] = 0.3 # lose
                        l[x][pi][str(int(x)+m)] = -(int(x) + m - int(x))
                        l[x][pi][str(int(x)-m)] = -(int(x) - m - int(x))

                    else: # bet on black
                        P[x][pi][str(int(x)+10*m)] = 0.2 # win
                        P[x][pi][str(int(x)-m)] = 0.8 # lose
                        l[x][pi][str(int(x)+10*m)] = -(int(x) + 10 * m - int(x))
                        l[x][pi][str(int(x)-m)] = -(int(x) - m - int(x))
                          
        return P, l
    
    def policy_eval(self, policy, max_iter=np.inf):
        """
        Policy Evaluation, given policy.
        """
        cts = 0
        while True:
            e = 0
            self.v_tmp = copy.deepcopy(self.V)

            for x in self.states:
                if int(x) > 0 and int(x) < 3:
                    v = self.V[x]
                    self.V[x] = self.value_cal(x, policy[x])
                    e = max([e, abs(v - self.V[x])])
            
            print('iter:', cts)
            print(self.V)          
            cts += 1
            if e < self.thresh or cts==max_iter:
                break
    
    
    def value_cal(self, x, policy):
        """
        Update Value function given policy.
        """
        v = 0
        for x_prime in self.P[x][policy]:
            p1 = self.P[x][policy][x_prime] * self.l[x][policy][x_prime]
            p2 = self.P[x][policy][x_prime] * self.gamma * self.v_tmp[x_prime]
            v += p1 + p2
#            v += self.P[x][policy][x_prime] * (self.l[x][policy][x_prime] \
#                       + self.gamma * self.V[x_prime])
        
        return v
    
    def policy_improve(self):
        """
        Policy improvement.
        """
        self.flag = True
        for x in self.states:
            if int(x) > 0 and int(x) < 3:
                p_tmp = self.policy[x]
                self.policy[x] = self.opt_policy(x)
                if p_tmp != self.policy[x]:
                    self.flag = False
#                    break
    
    def opt_policy(self, x):
        """
        Find the policy that make value function smallest.
        """
        v = np.inf
        for policy in self.U[x]:
            v_tmp = self.value_cal(x, policy)
            if v_tmp < v:
                v = v_tmp
                best_policy = policy
        
        return best_policy
    
    def policy_iter(self):
        """
        Policy Iteration.
        Initialize values to 0.
        """
        
        while True:
            self.policy_eval(self.policy)
            self.policy_improve()
            print("policy is", self.policy)
            if self.flag:
                break
            

def states_generator(x0):
    """
    Input: x0
    Generate all possible states.
    """
    states = [str(x0)]
    cts = 0
    n = len(states)
    while True:
        x = int(states[cts])
        if x > 0 and x < 3:
            for m in range(x + 1):
                if str(x+m) not in states:
                    states.append(str(x + m))
                if str(x-m) not in states:
                    states.append(str(x - m))
                if str(x+10*m) not in states:
                    states.append(str(x + 10*m))
        cts += 1
        if n < len(states):
            n = len(states)
        else:
            break
    for c, x in enumerate(states):
        x = int(x)
        states[c] = x
    states = np.sort(states)
    s = []
    for x in states:
        x = str(x)
        s.append(x)
    return s
            
    