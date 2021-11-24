import numpy as np
from numpy import random


def Aloha1(S, A):
    """Slotted Aloha 1-channel"""
    R = 0
    if sum(random.binomial(1, A, S)) == 1:
        Sp = S - 1
        R = 1
    else:
        Sp = S
    return Sp, R


timeHorizon = 1000
queueLength = 0
state = np.zeros(timeHorizon)
action = np.zeros(timeHorizon)
sp = 10

for t in range(timeHorizon):
    state[t] = sp
    action[t] = 1 / state[t]
    sp, reward = Aloha1(state[t], action[t])

    """Additional arrivals"""
    sp = sp + random.binomial(1,0.5)


