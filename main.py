from numpy import random
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


def run_sim1(S, A):
    """Slotted Aloha 1-channel"""
    R = 0
    if sum(random.binomial(1, A, S)) == 1:
        Sp = S - 1
        R = 1
    else:
        Sp = S
    return Sp, R


def run_simN(S, A, etaC):
    """Slotted Aloha N-channel"""
    R = 0
    num_active = sum(random.binomial(1, A, S))
    if num_active <= etaC:
        Sp = S - num_active
        R = num_active
    else:
        Sp = S
    return Sp, R


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
    Qold = np.zeros((num_S, num_A))
    V = np.zeros(num_S)

    for k in range(n_episode):
        """Define state, action, and reward with exploring start"""
        S = np.repeat(0, T)
        A = np.repeat(0.5, T)
        R = np.repeat(0, T)
        # S[0] = random.choice(S_space, p=S_space/sum(S_space))
        S[0] = 9
        A[0] = random.choice(A_space, p=np.repeat(1 / num_A, num_A))
        Sp, R[0] = run_sim1(S[0], A[0])

        """Generate an episode following given policy"""
        for t in range(1, T):
            S[t] = Sp
            iSt = np.where(S_space == S[t])[0].item()
            if random.uniform(0, 1, 1) > epsilon:
                A[t] = random.choice(A_space, p=policy[iSt])
            else:
                A[t] = random.choice(A_space, p=np.repeat(1 / num_A, num_A))
            Sp, R[t] = run_sim1(S[t], A[t])
            # R[t] = -run_simN(A[t], S[t], 2)
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

        """Convergence Check"""
        Qerr = np.max(np.abs(Q - Qold))
        if k % 200 == 0:
            print([k, Qerr])
            for i in range(num_S):
                V[i] = np.sum(Q[i] * policy[i])
            print(V)
            Qold = Q.copy()
            if Qerr < 0.001:
                break

    print(Qn)
    print(Q)
    print(policy)
    return Q, policy


def RL_TDTabular(n_episode, step_size, epsilon):
    """Define Variables"""
    S_space = np.arange(0, 10)
    A_space = 1 / np.arange(1, 10)
    num_S = len(S_space)
    num_A = len(A_space)
    policy = np.zeros(num_S)
    Q = np.zeros((num_S, num_A))
    Qold = np.zeros((num_S, num_A))
    V = np.zeros(num_S)
    T = 10000
    alpha = step_size
    """Discount Factor"""
    delta = 0

    for k in range(n_episode):
        """Define state, action, and reward with exploring start"""
        S = np.repeat(0, T)
        A = np.repeat(0.5, T)
        R = np.repeat(0, T)
        # S[0] = random.choice(S_space, p=S_space/sum(S_space))
        S[0] = 9

        for t in range(T - 1):
            """Epsilon-greedy"""
            iSt = np.where(S_space == S[t])[0].item()
            if random.uniform(0, 1, 1) > epsilon:
                iAt = np.argmax(Q[iSt])
            else:
                iAt = np.random.randint(num_A)
            A[t] = A_space[iAt]
            S[t+1], R[t] = run_sim1(S[t], A[t])
            # print('System:', S[t], A[t], R[t])

            """Q-Learning"""
            iSp = np.where(S_space == S[t+1])[0].item()
            iAp = np.argmax(Q[iSp])
            # print('Index:', iSt, iAt, iSp, iAp)
            Q[iSt, iAt] = (1 - alpha) * Q[iSt, iAt] + alpha * (R[t] + delta * Q[iSp, iAp])

            if S[t + 1] == 0:
                break
        # print(Q)

        """Convergence Check"""
        Qerr = np.max(np.abs(Q - Qold))
        if k % 5000 == 0 and k > 0:
            alpha = step_size/k*100000
            print('Update!')
            print([k, Qerr])

            """Update value and list of optimal actions"""
            for i in range(num_S):
                ia_star = np.argmax(Q[i])
                V[i] = Q[i, ia_star]
                policy[i] = A_space[ia_star]
            print(np.transpose(np.concatenate(([V],[policy]))))
            # print('Value Function', V)
            # print('Optimal Action', policy)
        Qold = Q.copy()
        #     # if Qerr < 0.001:
        #     #     break

    return Q, policy


if __name__ == "__main__":
    # Q, policy = RL_MonteCarloTabular(100000, 50, 0.3)
    Q, policy = RL_TDTabular(1000000, 0.0001, 0.3)
    print(Q)
    print(policy)
