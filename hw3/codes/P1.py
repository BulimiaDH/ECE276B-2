# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:40:31 2019

@author: epyir
"""

import numpy as np
from P1_utils import PI
from P1_utils import states_generator

def pb():
    states = states_generator(x0=1)
    pi_cal = PI(states)
    pi_cal.policy_eval(pi_cal.policy)
    
def pc():
    states = states_generator(x0=1)
    pi_cal = PI(states)
    pi_cal.policy_eval(pi_cal.policy, max_iter=1) # after just 1 iteration, crude evaluation
    pi_cal.policy_improve()
    print('Using crude evaluation the policy is ', pi_cal.policy)
    
    pi_cal = PI(states)
    pi_cal.policy_eval(pi_cal.policy) # after policy evaluation
    pi_cal.policy_improve()
    print('Using precise evaluation the policy is ', pi_cal.policy) # policy improvement for precise estimation

def pd():
    states = states_generator(x0=1)
    pi_cal = PI(states)
    pi_cal.policy_iter()
    print('Using precise evaluation the policy is ', pi_cal.policy) # policy improvement for precise estimation
    return pi_cal.policy
    
def simulation():
    policy = pd()
    # play this for 50000 times
    m = []
    for i in range(50000):
        print(i)
        x = 1
        while True:
            pi = policy[str(x)]
            prob = np.random.uniform(0, 1)
            if pi[0] == 0: # bet black
                if prob > 0.8: # win
                    x += pi[1] * 10
                else: # lose
                    x -= pi[1]
            elif pi[0] == 1: # bet red
                if prob < 0.3: # lose
                    x -= pi[1]
                else: # win
                    x += pi[1]
            
            if x <= 0 or x >= 3:
                m.append(x)
                break
    
    print("Expect to have:", np.mean(m))
        

if __name__ == '__main__':
    
    # test each problem
#    pb()
#    pc()
#    policy = pd()
    simulation()
    
