import numpy as np
import random

#  Constraints
#  
#  S[-1] = S[0] + diff
#  -> There are really only N-1 sites
#
#  Never  update S[-1] individually
#  Always update S[-1] at the end


import random

def reset_S(N, m):
    if m> 1: raise ValueError("Slope too big")
    if m<-1: raise ValueError("Slope too small")
    S = np.zeros(N)
    for i in range(1, N):
#         print(i)
#         print(S[i-1])
#         print(m*i/N)
#         print("\n")
        if (random.random() > (m/2)+.5) : S[i] = S[i-1] - 1
        else: S[i] = S[i-1] + 1
    return S

def site_update_S(S, site):
    N = len(S)
    if site < 0 or site > N-2:  raise ValueError("Bad site")
    if site == 0: 
        diff = S[-1] - S[0]
        return np.minimum(S[-2]+1-diff, S[1]+1)
    return     np.minimum(S[site-1]+1,  S[site+1]+1) 

def brick_update_S(S):
    print("USES WRONG UPDATING SCHEME")
    N = len(S)
    nextS = np.copy(S)
    for idx in range(N-1):
        nextS[idx] = site_update_S(S, idx)
        if nextS[idx] < S[idx]: raise ValueError("Decreasing entropy")
    nextS[-1] = nextS[0] + S[-1] - S[0]
    return(nextS)

def random_update_S(S):
    N = len(S)
    diff = S[-1] - S[0]
    i = random.randint(0,N-2)
#    print(i)
    S[i] = site_update_S(S, i)
    S[-1] = S[0] + diff
    return(S)

def stair_update_S(S, n):
    N = len(S)
    diff = S[-1] - S[0]
    i = random.randint(0,N-2)
#     print(i)
    for idx in range(n):
        i = i+1
        if i == N-1: i = 0
#         print(i)
        S[i] = site_update_S(S, i)
        S[-1] = S[0] + diff
    return(S)

def stair_update_S_fixed(S, n):
    N = len(S)
    i = random.randint(1,N-3)
#     print(i)
    for idx in range(n):
        i = i+1
        if i == N-1: break
#         print(i)
        S[i] = site_update_S(S, i)
    return(S)
