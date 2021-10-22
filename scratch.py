from numpy import random
import numpy as np


def run_sim1(p, num_user):
    outcome = 0
    if sum(random.binomial(1, p, num_user)) == 1:
        outcome = num_user - 1
    else:
        outcome = num_user
    return outcome


def RL_MonteCarlo1(T):
    S = np.repeat(0, T)
    A = np.repeat(0.0, T)
    R = np.repeat(0, T)
    S_space = np.arange(0, 10)
    A_space = 1 / np.arange(1, 20)
    num_S = len(S_space)
    num_A = len(A_space)
    policy = np.tile(np.repeat(1 / num_A, num_A), (num_S, 1))
    S[0] = random.choice(S_space, p=np.repeat(1 / num_S, num_S))
    A[0] = random.choice(A_space, p=np.repeat(1 / num_A, num_A))
    R[0] = -run_sim1(A[0], S[0])
    for t in range(1, T):
        S[t] = -R[t - 1]
        A[t] = random.choice(A_space, p=policy[np.where(S_space == S[t])[0].item()])
        R[t] = -run_sim1(A[t], S[t])

    return S, A, R


if __name__ == "__main__":
    # num_steps = 10
    # p = 0.1
    # num_user = 10
    # for i in range(num_steps):
    #     num_user = run_sim1(p, num_user)
    #     print(num_user)
    S, A, R = RL_MonteCarlo1(10)
    print(S)
    print(A)
