import numpy as np
from numpy import random
import itertools
import scipy.stats as stats


def sys_update(age, action):
    threshold_c = 3
    steps = 2
    age_max = 4
    lower, upper = 0, 3
    mu, sigma = 2, 0.5

    arrival = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    t1 = action[0]
    t2 = action[1]
    if sum(age[np.arange(t1,t2-1)]) <= threshold_c:
        reward = sum((age_max-np.arange(t1, t2-1))*age[np.arange(t1,t2-1)])
        # print(reward)
        age[np.arange(t1, t2-1)] = 0
    else:
        reward = 0
    # print(sum(age[-min(t2,steps):]))
    reward = reward - sum(age[-min(t2,steps):])
    age[steps:] = age[0:-steps]
    age[0:steps] = np.round(arrival.rvs(steps))

    # print(age, reward)
    return age, reward

def RL_MonteCarloTabular(n_episode, T, epsilon):
    threshold_c = 3
    steps = 2
    age_max = 4
    lower, upper = 0, 3

    """Define Variables"""
    S_space = np.array(list(itertools.product(*np.repeat([np.arange(0, upper+1)], age_max, axis=0))))
    A_space = np.array(list(itertools.combinations(range(1,age_max+2),2)))
    A_space[:,0] = A_space[:,0]-1
    num_S = len(S_space)
    num_A = len(A_space)

    policy = np.tile(np.repeat(1 / num_A, num_A), (num_S, 1))
    delta = 0.5
    Qn = np.zeros((num_S, num_A))
    Q = np.zeros((num_S, num_A))
    Qold = np.zeros((num_S, num_A))
    V = np.zeros(num_S)

    for k in range(n_episode):
        """Define state, action, and reward with exploring start"""
        S = np.repeat([np.zeros(age_max)], T, axis=0).astype(int)
        A = np.repeat([[0,2]], T, axis=0)
        R = np.repeat(0, T)
        S[0,:] = S_space[random.randint(0,num_S),:]
        A[0,:] = A_space[random.randint(0,num_A),:]
        Sp, R[0] = sys_update(S[0,:].copy(), A[0,:])
        # print(S[0,:])
        # print(A[0,:])
        # print(Sp)
        # print(R[0])

        """Generate an episode following given policy"""
        for t in range(1, T):
            S[t,:] = Sp
            iSt = np.where(S_space == S[t,:])[0].item()
            if random.uniform(0, 1, 1) > epsilon:
                A[t] = random.choice(A_space, p=policy[iSt])
            else:
                A[t] = random.choice(A_space, p=np.repeat(1 / num_A, num_A))
            Sp, R[t] = sys_update(S[t], A[t])
        print(S)

    #     """Monte Carlo prediction with Q-values"""
    #     SA = np.concatenate((S, A)).reshape((-1, 2), order='F')
    #     G = 0
    #     for t in range(T - 1, -1, -1):
    #         G = delta * G + R[t]
    #         if not any((SA[0:t - 1] == SA[t]).all(1)):
    #             iSt = np.where(S_space == S[t])[0].item()
    #             iAt = np.where(A_space == A[t])[0].item()
    #             Q[iSt][iAt] = (Q[iSt][iAt] * Qn[iSt][iAt] + G) / (Qn[iSt][iAt] + 1)
    #             Qn[iSt][iAt] += 1
    #
    #     """Policy improvement"""
    #     for i in range(num_S):
    #         a_star = np.amax(Q[i])
    #         ia_star = np.where(Q[i] == a_star)
    #         mask = np.zeros(num_A, dtype=bool)
    #         mask[ia_star] = True
    #         policy[i][mask] = 1 / len(ia_star[0])
    #         policy[i][~mask] = 0
    #
    #     """Convergence Check"""
    #     Qerr = np.max(np.abs(Q - Qold))
    #     if k % 200 == 0:
    #         print([k, Qerr])
    #         for i in range(num_S):
    #             V[i] = np.sum(Q[i] * policy[i])
    #         print(V)
    #         Qold = Q.copy()
    #         if Qerr < 0.001:
    #             break
    #
    # print(Qn)
    # print(Q)
    # print(policy)
    # return Q, policy

if __name__ == "__main__":
    # a,b = sys_update(np.array([3,2,1,1]),np.array([2,5]))
    # print(a,b)
    RL_MonteCarloTabular(1,10,1)
