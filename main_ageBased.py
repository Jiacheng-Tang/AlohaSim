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
REWARD = 4 * np.power(0.8 * np.ones(AGE_MAX), np.arange(AGE_MAX))
LOGPATH = './log/exp_age4_08/'


def sys_update(age, action):
    arrival = stats.truncnorm((ARRIVAL_MIN - MU) / SIGMA, (ARRIVAL_MAX - MU) / SIGMA, loc=MU, scale=SIGMA)
    t1 = action[0]
    t2 = action[1]
    if sum(age[np.arange(t1, t2 - 1)]) <= THRESHOLD:
        # print(sum(REWARD[np.arange(t1, t2 - 1)] * age[np.arange(t1, t2 - 1)]))
        reward = sum(REWARD[np.arange(t1, t2 - 1)] * age[np.arange(t1, t2 - 1)])
        age[np.arange(t1, t2 - 1)] = 0
    else:
        reward = 0
    # print(sum(age[-max(AGE_MAX - t2 + 1, STEPS):]))
    reward = reward - sum(age[-max(AGE_MAX - t2 + 1, STEPS):])
    age[t2 - 1:] = 0
    age[STEPS:] = age[0:-STEPS]
    age[0:STEPS] = np.round(arrival.rvs(STEPS))

    return age, reward


def RL_MonteCarloTabular(delta, n_episode, T, epsilon):
    """Define Variables"""
    S_space = np.array(list(itertools.product(*np.repeat([np.arange(0, ARRIVAL_MAX + 1)], AGE_MAX, axis=0))))
    A_space = np.array(list(itertools.combinations(range(1, AGE_MAX + 2), 2)))
    A_space[:, 0] = A_space[:, 0] - 1
    num_S = len(S_space)
    num_A = len(A_space)
    # print(A_space)
    # print(REWARD)

    policy = np.tile(np.repeat(1 / num_A, num_A), (num_S, 1))
    Qn = np.zeros((num_S, num_A))
    Q = np.zeros((num_S, num_A))
    Qold = np.zeros((num_S, num_A))
    V = np.zeros(num_S)

    for k in range(n_episode):
        """Define state, action, and reward with exploring start"""
        S = np.repeat([np.zeros(AGE_MAX)], T, axis=0).astype(int)
        A = np.repeat([[0, 2]], T, axis=0)
        R = np.repeat(0.0, T)
        S[0, :] = S_space[random.randint(0, num_S), :]
        A[0, :] = A_space[random.randint(0, num_A), :]
        Sp, R[0] = sys_update(S[0, :].copy(), A[0, :])
        # print(S[0, :], A[0, :], R[0])

        """Generate an episode following given policy"""
        for t in range(1, T):
            S[t, :] = Sp
            iSt = np.where(np.all(S_space == S[t, :], axis=1))[0].item()
            if random.uniform(0, 1, 1) > epsilon:
                iAt = random.choice(np.arange(num_A), p=policy[iSt])
                A[t, :] = A_space[iAt, :]
            else:
                A[t, :] = A_space[random.randint(0, num_A), :]
            Sp, R[t] = sys_update(S[t, :].copy(), A[t, :])
            # print(S[t, :], A[t, :], R[t])

        """Monte Carlo prediction with Q-values"""
        SA = np.concatenate((S, A), axis=1)
        # print(SA)
        G = 0
        for t in range(T - 1, -1, -1):
            G = delta * G + R[t]
            if not any(np.all((SA[0:t - 1, :] == SA[t, :]), axis=1)):
                iSt = np.where(np.all(S_space == S[t, :], axis=1))[0].item()
                iAt = np.where(np.all(A_space == A[t, :], axis=1))[0].item()
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

    header_A = ["--".join(items) for items in A_space.astype(str)]
    header_S = ["--".join(items) for items in S_space.astype(str)]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    Q_df = pd.DataFrame(Q, index=header_S, columns=header_A)
    Q_df.to_csv(LOGPATH + 'Q_' + timestr + '.csv', index=True, header=True)
    Qn_df = pd.DataFrame(Qn, index=header_S, columns=header_A)
    Qn_df.to_csv(LOGPATH + 'Qn_' + timestr + '.csv', index=True, header=True)
    V_df = pd.DataFrame(V, index=header_S)
    V_df.to_csv(LOGPATH + 'V_' + timestr + '.csv', index=True, header=False)
    Policy_df = pd.DataFrame(np.argmax(Q, axis=1), index=header_S, columns=None)
    Policy_df.to_csv(LOGPATH + 'Policy_' + timestr + '.csv', index=True, header=False)

    return


def AverageReward(policy):
    S_space = np.array(list(itertools.product(*np.repeat([np.arange(0, ARRIVAL_MAX + 1)], AGE_MAX, axis=0))))
    A_space = np.array(list(itertools.combinations(range(1, AGE_MAX + 2), 2)))
    A_space[:, 0] = A_space[:, 0] - 1
    # print(A_space)
    num_S = len(S_space)
    num_A = len(A_space)
    T = 100000

    S = np.repeat([np.zeros(AGE_MAX)], T, axis=0).astype(int)
    A = np.repeat([[0, 2]], T, axis=0)
    R = np.repeat(0, T)
    S[0, :] = S_space[random.randint(0, num_S), :]
    A[0, :] = A_space[random.randint(0, num_A), :]
    Sp, R[0] = sys_update(S[0, :].copy(), A[0, :])

    for t in range(1, T):
        S[t, :] = Sp
        iSt = np.where(np.all(S_space == S[t, :], axis=1))[0].item()
        A[t, :] = A_space[policy[iSt], :]
        Sp, R[t] = sys_update(S[t, :].copy(), A[t, :])
        if t % 10000 == 0:
            print(t)

    print(np.mean(R))
    return


def StepReward(policy):
    S_space = np.array(list(itertools.product(*np.repeat([np.arange(0, ARRIVAL_MAX + 1)], AGE_MAX, axis=0))))
    A_space = np.array(list(itertools.combinations(range(1, AGE_MAX + 2), 2)))
    A_space[:, 0] = A_space[:, 0] - 1
    num_S = len(S_space)
    reward = np.zeros(num_S)
    for i in range(num_S):
        reward[i] = sys_update(S_space[i, :].copy(), A_space[policy[i], :])[1]

    print(reward)
    return reward


if __name__ == "__main__":
    RL_MonteCarloTabular(0.1, 100000, 100, 0.3)
    RL_MonteCarloTabular(0.5, 100000, 100, 0.3)
    RL_MonteCarloTabular(0.8, 100000, 100, 0.3)

    # df01 = pd.read_csv('./log/Policy_20211229-052255.csv', header=None, index_col=0)
    # p01 = df01.to_numpy().flatten()
    # r1 = StepReward(p01)
    #
    # df05 = pd.read_csv('./log/Policy_20211229-070830.csv', header=None, index_col=0)
    # p05 = df05.to_numpy().flatten()
    #
    # df08 = pd.read_csv('./log/Policy_20211229-085407.csv', header=None, index_col=0)
    # p08 = df08.to_numpy().flatten()
    # r2 = StepReward(p08)
    # print(r1 == r2)
    #
    # AverageReward(p01)
    # AverageReward(p05)
    # AverageReward(p08)

    # df01 = pd.read_csv('./log/Policy_20211229-142856.csv', header=None, index_col=0)
    # p01 = df01.to_numpy().flatten()
    #
    # df05 = pd.read_csv('./log/Policy_20211229-172003.csv', header=None, index_col=0)
    # p05 = df05.to_numpy().flatten()
    #
    # df08 = pd.read_csv('./log/Policy_20211229-201243.csv', header=None, index_col=0)
    # p08 = df08.to_numpy().flatten()
    # AverageReward(p01)
    # AverageReward(p05)
    # AverageReward(p08)
