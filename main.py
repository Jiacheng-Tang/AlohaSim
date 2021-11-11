from numpy import random
import numpy as np


def run_sim1(p, num_user):
    """Slotted Aloha 1-channel"""
    outcome = 0
    if sum(random.binomial(1, p, num_user)) == 1:
        outcome = num_user - 1
    else:
        outcome = num_user
    return outcome


def run_simN(p, num_user,etaC):
    """Slotted Aloha N-channel"""
    outcome = 0
    num_active = sum(random.binomial(1, p, num_user))
    if num_active <= etaC:
        outcome = num_user - num_active
    else:
        outcome = num_user
    return outcome


def RL_MonteCarloTabular(n_episode, T, epsilon):
    """Define Variables"""
    S_space = np.arange(0, 10)
    A_space = 1 / np.arange(1, 15)
    num_S = len(S_space)
    num_A = len(A_space)
    policy = np.tile(np.repeat(1 / num_A, num_A), (num_S, 1))
    delta = 0.5
    Qn = np.zeros((num_S, num_A))
    Q = np.zeros((num_S, num_A))

    for k in range(n_episode):
        """Define state, action, and reward with exploring start"""
        S = np.repeat(0, T)
        A = np.repeat(0.5, T)
        R = np.repeat(0, T)
        # S[0] = random.choice(S_space, p=np.repeat(1 / num_S, num_S))
        S[0] = random.choice(S_space, p=S_space/sum(S_space))
        A[0] = random.choice(A_space, p=np.repeat(1 / num_A, num_A))
        # R[0] = -run_sim1(A[0], S[0])
        R[0] = -run_simN(A[0], S[0], 2)

        """Generate an episode following given policy"""
        for t in range(1, T):
            S[t] = -R[t - 1]
            iSt = np.where(S_space == S[t])[0].item()
            if random.uniform(0,1,1)>epsilon:
                A[t] = random.choice(A_space, p=policy[iSt])
            else:
                A[t] = random.choice(A_space, p=np.repeat(1 / num_A, num_A))
            # R[t] = -run_sim1(A[t], S[t])
            R[t] = -run_simN(A[t], S[t], 2)
            if S[t] == 0:
                break

        """Monte Carlo prediction with Q-values"""
        SA = np.concatenate((S, A)).reshape((-1, 2), order='F')
        G = 0
        for t in range(T - 1, -1, -1):
            G = delta * G + R[t]
            if not any((SA[0:t - 1] == SA[t]).all(1)):
                iSt = np.where(S_space == S[t])[0].item()
                iAt = np.where(A_space == A[t])[0].item()
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

    print(Qn)
    return Q, policy


if __name__ == "__main__":
    Q, policy = RL_MonteCarloTabular(10000, 50, 0.3)
    # print(Q)
    print(policy)
