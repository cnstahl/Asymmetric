import numpy as np
import random

#  Constraints
#  




def reset_S(N, m):
    if m> 1: raise ValueError("Slope too big")
    if m<-1: raise ValueError("Slope too small")
    S = np.zeros(N)
    for i in range(1, N):
#         print(i)
#         print(S[i-1])
#         print(m*i/N)
#         print("\n")
        if S[i-1] > (m*i): S[i] = S[i-1] - 1
        else: S[i] = S[i-1] + 1
    return S

def brick_update_S(S):
    N = len(S)
    diff = S[-1] - S[0]
    nextS = np.copy(S)
    nextS[0] = np.minimum(S[-1]+1-diff, S[1]+1)
    for idx, val in enumerate(S):
        if (idx != 0) and (idx != N-1):
            nextS[idx] = np.minimum(S[idx-1]+1, S[idx+1]+1)
            if nextS[idx] < S[idx]: raise ValueError("Decreasing entropy")
    nextS[-1] = nextS[0] + diff
    return(nextS)

def random_update_S(S):
    N = len(S)
    diff = S[-1] - S[0]
    i = random.randint(0,N-2)
#     print(i)
    if i == 0: S[0] = np.minimum(S[ -2]+1-diff, S[1]+1)
    else:      S[i] = np.minimum(S[i-1]+1,      S[i+1]+1)
    S[-1] = S[0] + diff
    return(S)

