from numpy import random
import numpy as np
from main import *
from nn import *
import torch


def RL_MonteCarloNN(n_episode, T, epsilon):
    """Define Variables"""
    S_space = np.arange(0, 10)
    S_max = max(S_space)
    num_S = len(S_space)
    policy = np.zeros(num_S)
    for i in range(1,num_S):
        policy[i] = random.uniform(0,1/i)
    # print(policy)
    delta = 0.5

    """Define neural network (normalized input)"""
    NN_PARAMETER = {
        'input_dim': 2,
        'layers': 4,
        'width': 4,
        'lr': 0.01,
        'positive': False,
        'epochs': 5000,
        'tol': 1e-4,
        'input_range': (0, 1)
    }
    Qnn = NeuralNetwork(NN_PARAMETER)

    for k in range(n_episode):
        """Define state, action, and reward with exploring start"""
        S = np.repeat(0, T)
        A = np.repeat(0.5, T)
        R = np.repeat(0, T)
        # S[0] = random.choice(S_space, p=np.repeat(1 / num_S, num_S))
        S[0] = S_max
        A[0] = random.uniform(0, 1, 1)
        R[0] = -run_sim1(A[0], S[0])

        """Generate an episode following given policy"""
        for t in range(1, T):
            S[t] = -R[t - 1]
            iSt = np.where(S_space == S[t])[0].item()
            if random.uniform(0,1,1)>epsilon:
                A[t] = policy[iSt]
            else:
                A[t] = random.uniform(0, 1, 1)
            R[t] = -run_sim1(A[t], S[t])
            if S[t] < 1:
                break

        """Monte Carlo prediction with Q-values"""
        SA = np.concatenate((S, A)).reshape((-1, 2), order='F')
        print(SA)
        G = 0
        NNin = np.empty((0,2))
        NNout = np.empty((0,1))
        for t in range(T - 1, -1, -1):
            G = delta * G + R[t]
            if not any((SA[0:t] == SA[t]).all(1)):
                NNin = np.append(NNin, [SA[t]], axis=0)
                NNout = np.append(NNout, [G])
        NNin = torch.from_numpy(NNin).type(torch.float32).to('cuda')
        NNin[:,0] = NNin[:,0]/S_max
        NNout = torch.from_numpy(NNout).type(torch.float32).to('cuda')
        print(NNin)
        print(NNout)
        Qnn.train(NNin,NNout)


        """Policy improvement"""
        # for i in range(num_S):
        #     a_star = np.amax(Q[i])
        #     ia_star = np.where(Q[i] == a_star)
        #     mask = np.zeros(num_A, dtype=bool)
        #     mask[ia_star] = True
        #     policy[i][mask] = 1 / len(ia_star[0])
        #     policy[i][~mask] = 0
    return policy

if __name__ == "__main__":
    policy = RL_MonteCarloNN(5, 1000, 0.1)
    # print(Q)
    # print(policy)
