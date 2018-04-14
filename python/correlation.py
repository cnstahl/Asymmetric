import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats
from growthrates import *

def nth_correlator(S, n):
    N = len(S)
    m = (S[-1] - S[0])/(N-1)
    Cij = 0
    for i in range(n):
        Cij += (S[i-n] - S[i-1-n]) * (S[i+1] - S[i])
    for i in range(1, N-n):
        Cij += (S[i] - S[i-1]) * (S[i+n] - S[i+n-1])
    return Cij/(N-1)

def get_correlator(S):
    N = len(S)
    m = (S[-1] - S[0])/(N-1)
    Cij = (S[-1] - S[-2]) * (S[1] - S[0])
    for i in range(1, N-1):
        Cij += (S[i] - S[i-1]) * (S[i+1] - S[i])
    return Cij/(N-1)

N      = 101
init   = 400
n      = 1
steps  = 100
trials = 1000
Nms    = 21
finalCors1   = np.zeros((trials, Nms))
ms = np.linspace(-1, 1, Nms)
for i in range(Nms):
    S = reset_S(N, ms[i])
    for j in range(init):
        S = stair_update_S(S, n)
    for j in range(trials):
        for k in range(steps):
            S = stair_update_S(S, n)
        finalCors1[j, i] = get_correlator(S) - ms[i]**2

n = 2
finalCors2   = np.zeros((trials, Nms))
for i in range(Nms):
    S = reset_S(N, ms[i])
    for j in range(init):
        S = stair_update_S(S, n)
    for j in range(trials):
        for k in range(steps):
            S = stair_update_S(S, n)
        finalCors2[j, i] = get_correlator(S) - ms[i]**2

n = 3
finalCors3   = np.zeros((trials, Nms))
for i in range(Nms):
    S = reset_S(N, ms[i])
    for j in range(init):
        S = stair_update_S(S, n)
    for j in range(trials):
        for k in range(steps):
            S = stair_update_S(S, n)
        finalCors3[j, i] = get_correlator(S) - ms[i]**2

plt.errorbar(ms, np.mean(finalCors3, axis=0), yerr=np.std(finalCors3, axis=0), label="3-Stairs", fmt='-o')
plt.errorbar(ms, np.mean(finalCors2, axis=0), yerr=np.std(finalCors2, axis=0), label="2-Stairs", fmt='-o')
plt.errorbar(ms, np.mean(finalCors1, axis=0), yerr=np.std(finalCors1, axis=0), label="1-Stairs", fmt='-o')
plt.plot(ms, .15-.15*(ms)**2, label="Analytic")
plt.legend(loc=2)
plt.xlabel("Slope")
plt.ylabel("Correlation")
plt.title("Initial and Final Correlations in 3-stair Circuit")
plt.savefig("../figures/correlations/goodstairCorrel.pdf")
