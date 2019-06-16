# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:18:54 2019

@author: epyir
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:25:46 2019

@author: epyir
"""

import numpy as np
from matplotlib import pyplot as plt

        
def PRS_generator(n = 100, prefer = "R"):
    """
        Use for generator a sequence of Rock-Paper-Scissors based on preference.
        Based on problem 5 preference owns 50% probability.
        Inputs:
            n - length of sequence
            prefer - "R", "P" or "S"
        Output:
            sequence - a sequence of plays
    """
    
    choice = ["R", "S", "P"]
    choice.remove(prefer) # remove the preference
    Sq = np.zeros((n))
    np.random.seed(10) # this seed is for re-run the program
    random_array = np.random.uniform(0, 1, n) # uniformly draw
    Sq[np.logical_and(random_array >= 0., random_array < 0.5)] = 1
    Sq[np.logical_and(random_array >= 0.5, random_array < 0.75)] = 0
    Sq[np.logical_and(random_array >= 0.75, random_array < 1.)] = -1
    Sq = Sq.astype(int)
    Sequence = []
    
    # transfer num to string {"R","P","S"}
    for i in range(n):
        if Sq[i] == 1:
            Sequence.append(prefer)
        if Sq[i] == 0:
            Sequence.append(choice[0])
        if Sq[i] == -1:
            Sequence.append(choice[1])
    
    return Sequence

def PRS_str2num(PRS_str):
    """
        transfer string to integer for convenience.
    """
    
    if PRS_str == "R":
        PRS_num = 0
    if PRS_str == "P":
        PRS_num = 1
    if PRS_str == "S":
        PRS_num = 2
        
    return int(PRS_num)

def PRS_judge(play1, play2, score = 0, output_result = '0'):
    """
        Inputs: 
            play1 - player 1's play
            play2 - player 2's play
            both arguements should be in {"R", "P", "S"}
            score - the difference between player 1 and 2 (score of player 1)
            output_result - '0': not output result of this round
                            else: output result of this round
        Outputs:
            score - the difference between player 1 and 2 after jugdement
    """
    # score matrix
    #   | R| P| S
    # R | 0|-1| 1
    # P | 1| 0|-1
    # S |-1| 1| 0
    score_mat = np.array([[0, -1, 1],
                          [1, 0, -1],
                          [-1, 1, 0]]) # score matrix
    score_t = score_mat[PRS_str2num(play1), PRS_str2num(play2)] # decide the score at time t
    score += score_t # cumulate
    
    # decide whether output score at time t
    if output_result == '0': 
        return score
    else: 
        return score, score_t

def Strategy(nt, style, lastplay):
    """
        Inputs:
            style - how to play in {"DP", "Stochastic", "Deterministic"}
            nt - an array of #["R","P","S"], 1x3
            lastplay - last round play
        Output:
            Optimal policy
    """
    # using different style to play
    if style == "DP":
        policy = DP_optimal(nt)
        return policy
    if style == "Stochastic":
        draw = np.random.uniform(0, 1) # uniformly draw
        if 0 <= draw and draw < 1/3:
            return "R"
        if 1/3 <= draw and draw < 2/3:
            return "P"
        if 2/3 <= draw and draw < 1:
            return "S"
    if style == "Deterministic":
        if lastplay == "S":
            return "R"
        if lastplay == "R":
            return "P"
        if lastplay == "P":
            return "S"

def DP_optimal(nt):
    """
        Compute Expectation wrt each control input given z(1:t)
        Inputs: 
            nt - an array of #["R","P","S"]
            n - # rounds
        Output:
            policy - optimal policy
    """

    exp_dict = {'0':'R', '1': 'P', '2': 'S'} # use a dictionary to store 
    
    # freq2prob
    bt = ml_freq2prob(nt)
    
    # compute the expectation given different control input
    exp_R = bt[0] * PRS_judge("R", "R") + bt[1] * PRS_judge("R", "P") + bt[2] * PRS_judge("R", "S")
    exp_P = bt[0] * PRS_judge("P", "R") + bt[1] * PRS_judge("P", "P") + bt[2] * PRS_judge("P", "S")
    exp_S = bt[0] * PRS_judge("S", "R") + bt[1] * PRS_judge("S", "P") + bt[2] * PRS_judge("S", "S")

    # choose the maximum
    idx = np.argmax([exp_R, exp_P, exp_S])
    return exp_dict[str(idx)]

def ml_freq2prob(nt):
    nt = nt.reshape(3,)
    bt = np.zeros((3))
    # update based on Bayes Rule, see report for details
    p_rps = 0.5**nt[0] * 0.25**nt[1] * 0.25**nt[2] * 1/3 +\
            0.25**nt[0] * 0.5**nt[1] * 0.25**nt[2] * 1/3+ \
            0.25**nt[0] * 0.25**nt[1] * 0.5**nt[2] * 1/3
    bt[0] = (0.5 * (0.5**(nt[0]) * 0.25**(nt[1]) * 0.25**(nt[2])) * 1/3 \
            + 0.25 * (0.25**(nt[0]) * 0.5**(nt[1]) * 0.25**(nt[2])) * 1/3 \
            + 0.25 * (0.25**(nt[0]) * 0.25**(nt[1]) * 0.5**(nt[2])) * 1/3)/p_rps
    bt[1] = (0.25 * (0.5**(nt[0]) * 0.25**(nt[1]) * 0.25**(nt[2])) * 1/3 \
            + 0.5 * (0.25**(nt[0]) * 0.5**(nt[1]) * 0.25**(nt[2])) * 1/3 \
            + 0.25 * (0.25**(nt[0]) * 0.25**(nt[1]) * 0.5**(nt[2])) * 1/3)/p_rps
    bt[2] = (0.25 * (0.5**(nt[0]) * 0.25**(nt[1]) * 0.25**(nt[2])) * 1/3 \
            + 0.5 * (0.25**(nt[0]) * 0.5**(nt[1]) * 0.25**(nt[2])) * 1/3 \
            + 0.25 * (0.25**(nt[0]) * 0.25**(nt[1]) * 0.5**(nt[2])) * 1/3)/p_rps
    
    return nt/np.sum(nt)

def nt_update(nt, play):
    """
        update nt
        Inputs:
            nt - # of R, P, S. 1x3
            play - opponent's play in this round
    """
    plus = PRS_str2num(play)
    nt[plus] += 1
    return nt


if __name__=="__main__":
    
    T = 100
    N = 50
    player_2_TN = PRS_generator(T * N, "P")
    player_2_TN = np.random.permutation(player_2_TN)

    score_DP_TN = np.zeros((N, T))
    score_ST_TN = np.zeros((N, T))
    score_DE_TN = np.zeros((N, T))
    
    # Forward DP algorithm
    for j in range(N):        
        # generate opponent's play
        player_2 = player_2_TN[j * T: (j + 1) * T]
        x_t_last = 'S' # for deterministic play style
        
        # score initialization
        score_DP = 0 
        score_ST = 0
        score_DE = 0

        # belief intialization
        bt = np.array([1/3, 1/3, 1/3])
        nt = np.array([0, 0, 0])
        
        
        for i in range(T):
            xt = player_2[i]
            
            style = 'DP' # choose style: DP
            ut = Strategy(nt, style, x_t_last) # use related style to choose strategy
            score_DP, rlt = PRS_judge(ut, xt, score_DP, '1') # judge the score this round
            score_DP_TN[j, i] = score_DP # record score for statistical purpose
            nt = nt_update(nt, xt) # update nt
            bt = ml_freq2prob(nt) # update bt
            
            style = 'Stochastic'
            ut = Strategy(nt, style, x_t_last)
            score_ST, rlt = PRS_judge(ut, xt, score_ST, '1')
            score_ST_TN[j, i] = score_ST

            
            style = 'Deterministic'
            ut = Strategy(nt, style, x_t_last)
            score_DE, rlt = PRS_judge(ut, xt, score_DE, '1')
            score_DE_TN[j, i] = score_DE
            x_t_last = ut  # record last time play to determine next time
    
    # compute mean and std of three play style
    score_mean_DP = np.mean(score_DP_TN, axis=0)
    score_std_DP = np.std(score_DP_TN, axis=0)
    
    score_mean_ST = np.mean(score_ST_TN, axis=0)
    score_std_ST = np.std(score_ST_TN, axis=0)
    
    score_mean_DE = np.mean(score_DE_TN, axis=0)
    score_std_DE = np.std(score_DE_TN, axis=0)
    
    # display the result
    fig = plt.figure()
    plt.plot(score_mean_DP)
    plt.plot(score_mean_ST)
    plt.plot(score_mean_DE)
    plt.legend(["DP","Stochastic","Deterministic"])
    plt.xlabel("Rounds")
    plt.ylabel("Score Differential")
    plt.title("Mean Score along the Rounds")
    plt.grid()
    plt.show()

    fig = plt.figure()
    plt.plot(score_std_DP)
    plt.plot(score_std_ST)
    plt.plot(score_std_DE)
    plt.legend(["DP","Stochastic","Deterministic"])
    plt.xlabel("Rounds")
    plt.ylabel("Score Differential")
    plt.title("Standard Deviation along the Rounds")
    plt.grid()
    plt.show()  
    