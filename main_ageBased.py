import numpy as np
from numpy import random
import itertools
import scipy.stats as stats
import pandas as pd
import time

"""Define Variables"""
AGE_MAX, ARRIVAL_MIN, ARRIVAL_MAX = 4, 0, 3
THRESHOLD, STEPS = 3, 2
MU, SIGMA = 2, 0.5
DELTA = 0.8

def sys_update(age, action):
    arrival = stats.truncnorm((ARRIVAL_MIN - MU) / SIGMA, (ARRIVAL_MAX - MU) / SIGMA, loc=MU, scale=SIGMA)
    t1 = action[0]
    t2 = action[1]
    if sum(age[np.arange(t1,t2-1)]) <= THRESHOLD:
        reward = sum((AGE_MAX-np.arange(t1, t2-1))*age[np.arange(t1,t2-1)])
        # print(reward)
        age[np.arange(t1, t2-1)] = 0
    else:
        reward = 0
    # print(sum(age[-min(t2,steps):]))
    reward = reward - sum(age[-min(t2,STEPS):])
    age[STEPS:] = age[0:-STEPS]
    age[0:STEPS] = np.round(arrival.rvs(STEPS))

    # print(age, reward)
    return age, reward

def RL_MonteCarloTabular(n_episode, T, epsilon):
    """Define Variables"""
    S_space = np.array(list(itertools.product(*np.repeat([np.arange(0, ARRIVAL_MAX+1)], AGE_MAX, axis=0))))
    A_space = np.array(list(itertools.combinations(range(1,AGE_MAX+2),2)))
    A_space[:,0] = A_space[:,0]-1
    num_S = len(S_space)
    num_A = len(A_space)

    policy = np.tile(np.repeat(1 / num_A, num_A), (num_S, 1))
    Qn = np.zeros((num_S, num_A))
    Q = np.zeros((num_S, num_A))
    Qold = np.zeros((num_S, num_A))
    V = np.zeros(num_S)

    for k in range(n_episode):
        """Define state, action, and reward with exploring start"""
        S = np.repeat([np.zeros(AGE_MAX)], T, axis=0).astype(int)
        A = np.repeat([[0,2]], T, axis=0)
        R = np.repeat(0, T)
        S[0,:] = S_space[random.randint(0,num_S),:]
        A[0,:] = A_space[random.randint(0,num_A),:]
        Sp, R[0] = sys_update(S[0,:].copy(), A[0,:])
        # print(S[0,:], A[0,:], R[0])

        """Generate an episode following given policy"""
        for t in range(1, T):
            S[t,:] = Sp
            iSt = np.where(np.all(S_space == S[t,:],axis=1))[0].item()
            if random.uniform(0, 1, 1) > epsilon:
                iAt = random.choice(np.arange(num_A), p=policy[iSt])
                A[t,:] = A_space[iAt,:]
            else:
                A[t,:] = A_space[random.randint(0,num_A),:]
            Sp, R[t] = sys_update(S[t,:].copy(), A[t,:])
            # print(S[t, :], A[t, :], R[t])

        """Monte Carlo prediction with Q-values"""
        SA = np.concatenate((S, A), axis=1)
        # print(SA)
        G = 0
        for t in range(T - 1, -1, -1):
            G = DELTA * G + R[t]
            if not any(np.all((SA[0:t - 1,:] == SA[t,:]),axis=1)):
                iSt = np.where(np.all(S_space == S[t,:],axis=1))[0].item()
                iAt = np.where(np.all(A_space == A[t,:],axis=1))[0].item()
                Q[iSt][iAt] = (Q[iSt][iAt] * Qn[iSt][iAt] + G) / (Qn[iSt][iAt] + 1)
                Qn[iSt][iAt] += 1

        """Policy improvement"""
        for i in range(num_S):
            a_star = np.amax(Q[i])
            ia_star = np.where(Q[i] == a_star)
            mask = np.zeros(num_A, dtype=bool)
            mask[ia_star] = True
            policy[i][mask] = 1 / len(ia_star[0])
            policy[i][~mask] = 0

        """Convergence Check"""
        Qerr = np.max(np.abs(Q - Qold))
        if k % 100 == 0:
            print([k, Qerr])
            for i in range(num_S):
                V[i] = np.sum(Q[i] * policy[i])
            # print(V)
            Qold = Q.copy()
            if Qerr < 0.001:
                break
    # print(A_space)

    header_A = ["--".join(items) for items in A_space.astype(str)]
    header_S = ["--".join(items) for items in S_space.astype(str)]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    Q_df = pd.DataFrame(Q, index=header_S, columns=header_A)
    Q_df.to_csv('./log/Q_'+timestr+'.csv', index=True, header=True)
    Qn_df = pd.DataFrame(Qn, index=header_S, columns=header_A)
    Qn_df.to_csv('./log/Qn_'+timestr+'.csv', index=True, header=True)
    V_df = pd.DataFrame(V, index=header_S)
    V_df.to_csv('./log/V_'+timestr+'.csv', index=True, header=False)

    return

if __name__ == "__main__":
    RL_MonteCarloTabular(100000,100,0.3)
