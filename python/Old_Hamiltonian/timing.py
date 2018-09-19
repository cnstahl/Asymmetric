import numpy as np
import scipy.linalg as la
import hamiltonian as hm
import matplotlib.pyplot as plt
from random import gauss

def rand_vector():
    vec = [gauss(0, 1) for i in range(3)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def SU2breaker(L):
    H = np.zeros((2**L,2**L), dtype=complex)
    for j in range(L//2):
        hs = []
        for i in range(L):
            hs.append(np.eye(2,2, dtype=complex))
        vect = rand_vector()
        h = vect[0]*hm.X + vect[1]*hm.Y + vect[2]*hm.Z
        hs[j] = h
        H += hm.tensor(*hs)
        hs[j] = np.eye(2,2, dtype=complex)
        hs[L-j-1] = np.zeros((2,2), dtype=complex) - h
        H += hm.tensor(*hs)
    return H

for L in range(11,12):
    H  = hm.dense_H(L) + .001*SU2breaker(L)
    valsH, vecsH = la.eigh(H)
    print(L, np.sum(np.isclose(valsH,0)))
