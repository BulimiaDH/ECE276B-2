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
    choice.remove(prefer)
    Sq = np.zeros((n))
    np.random.seed(10) # this seed is for re-run the program
    random_array = np.random.uniform(0, 1, n)
    Sq[np.logical_and(random_array >= 0., random_array < 0.5)] = 1
    Sq[np.logical_and(random_array >= 0.5, random_array < 0.75)] = 0
    Sq[np.logical_and(random_array >= 0.75, random_array < 1.)] = -1
    Sq = Sq.astype(int)
    Sequence = []
    
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
                          [-1, 1, 0]])
    score_t = score_mat[PRS_str2num(play1), PRS_str2num(play2)]
    score += score_t
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
    # score matrix
    #   | R| P| S
    # R | 0|-1| 1
    # P | 1| 0|-1
    # S |-1| 1| 0
#    score_mat = np.array([[0, -1, 1],
#                          [1, 0, -1],
#                          [-1, 1, 0]])
    exp_dict = {}
    
    # freq2prob
    bt = ml_freq2prob(nt)
    
    # compute the expectation given different control input
    exp_R = bt[0] * PRS_judge("R", "R") + bt[1] * PRS_judge("R", "P") + bt[2] * PRS_judge("R", "S")
    exp_P = bt[0] * PRS_judge("P", "R") + bt[1] * PRS_judge("P", "P") + bt[2] * PRS_judge("P", "S")
    exp_S = bt[0] * PRS_judge("S", "R") + bt[1] * PRS_judge("S", "P") + bt[2] * PRS_judge("S", "S")
    exp_dict = {'0':'R', '1': 'P', '2': 'S'}
    # choose the maximum
    idx = np.argmax([exp_R, exp_P, exp_S])
    return exp_dict[str(idx)]

def ml_freq2prob(nt):
    return nt/np.sum(nt)

def nt_update(nt, play):
    """
        update nt
    """
    plus = PRS_str2num(play)
    nt[plus] += 1
    return nt


if __name__=="__main__":
    
    T = 100
    N = 50
    player_2_TN = PRS_generator(T * N, "P")
    player_2_TN = np.random.permutation(player_2_TN)

    
#    score_N_round = np.zeros((3, N))
    score_DP_TN = np.zeros((N, T))
    score_ST_TN = np.zeros((N, T))
    score_DE_TN = np.zeros((N, T))
    
    # Forward DP algorithm
    for j in range(N):
        print(j)
        player_2 = player_2_TN[j * T: (j + 1) * T]
        x_t_last = 'S'
        score_DP = 0
        score_ST = 0
        score_DE = 0

#        player_1_DP = []
#        player_1_ST = []
#        player_1_DE = []

        bt = np.array([1/3, 1/3, 1/3])
        nt = np.array([0, 0, 0])
        for i in range(T):
            xt = player_2[i]
            
            style = 'DP' # choose style
            ut = Strategy(nt, style, x_t_last) # use related style to choose strategy
#            player_1_DP.append(ut) # record
            score_DP, rlt = PRS_judge(ut, xt, score_DP, '1') # judge the score this round
            score_DP_TN[j, i] = score_DP # record score for statistical purpose
#            results.append(rlt)
            nt = nt_update(nt, xt) # update nt
            bt = ml_freq2prob(nt) # update bt
            
            style = 'Stochastic'
            ut = Strategy(nt, style, x_t_last)
#            player_1_ST.append(ut)
            score_ST, rlt = PRS_judge(ut, xt, score_ST, '1')
            score_ST_TN[j, i] = score_ST
#            results.append(rlt)
            
            style = 'Deterministic'
            ut = Strategy(nt, style, x_t_last)
#            player_1_DE.append(ut)
            score_DE, rlt = PRS_judge(ut, xt, score_DE, '1')
            score_DE_TN[j, i] = score_DE
#            results.append(rlt)
            x_t_last = ut  # record last time play to determine next time
        
#        score_N_round[0, j] = score_DP
#        score_N_round[1, j] = score_ST
#        score_N_round[2, j] = score_DE
        
    score_mean_DP = np.mean(score_DP_TN, axis=0)
    score_std_DP = np.std(score_DP_TN, axis=0)
    
    score_mean_ST = np.mean(score_ST_TN, axis=0)
    score_std_ST = np.std(score_ST_TN, axis=0)
    
    score_mean_DE = np.mean(score_DE_TN, axis=0)
    score_std_DE = np.std(score_DE_TN, axis=0)
    
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
    