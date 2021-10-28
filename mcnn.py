from numpy import random
import numpy as np
from main import *
from nn import *


def RL_MonteCarloNN(n_episode, T):
    """Define Variables"""
    S_space = np.arange(0, 10)
    num_S = len(S_space)
    policy = random.uniform(0, 1, num_S)
    delta = 0.5

    """Define neural network (normalized input)"""
    NN_PARAMETER = {
        'input_dim': 2,
        'layers': 8,
        'width': 1,
        'lr': 0.01,
        'positive': False,
        'epochs': T,
        'tol': 1e-4,
        'input_range': (0, 1)
    }
    Qnn = NeuralNetwork(NN_PARAMETER)

    for k in range(n_episode):
        """Define state, action, and reward with exploring start"""
        S = np.repeat(0, T)
        A = np.repeat(0.5, T)
        R = np.repeat(0, T)
        S[0] = random.choice(S_space, p=np.repeat(1 / num_S, num_S))
        A[0] = random.uniform(0, 1, 1)
        R[0] = -run_sim1(A[0], S[0])

        """Generate an episode following given policy"""
        for t in range(1, T):
            S[t] = -R[t - 1]
            iSt = np.where(S_space == S[t])[0].item()
            A[t] = policy[iSt]
            R[t] = -run_sim1(A[t], S[t])
            if S[t] == 0:
                break

        """Monte Carlo prediction with Q-values"""
        SA = np.concatenate((S, A)).reshape((-1, 2), order='F')
        G = 0
        NNin = np.empty((0,3))
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
    return Q, policy