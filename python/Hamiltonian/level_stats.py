import numpy as np
import scipy.linalg as la
import hamiltonian as hm

L = 13
H = hm.sparse_H(L)

Z = np.zeros((2**L,2**L))
for i in range(L):
    Zi = 1
    for j in range(L):
        if (j==i): Zi = np.kron(Zi,hm.Z)
        else: Zi = np.kron(Zi,hm.I)
    Z += Zi

X = np.zeros((2**L,2**L))
for i in range(L):
    Xi = 1
    for j in range(L):
        if (j==i): Xi = np.kron(Xi,hm.X)
        else: Xi = np.kron(Xi,hm.I)
    X += Xi

Y = np.zeros((2**L,2**L), dtype=complex)
for i in range(L):
    Yi = 1
    for j in range(L):
        if (j==i): Yi = np.kron(Yi,hm.Y)
        else: Yi = np.kron(Yi,hm.I)
    Y += Yi
S2 = X@X + Y@Y + Z@Z

Hsmal = hm.mat2list(H)[(int)(L/2)]
Zsmal = hm.mat2list(Z)[(int)(L/2)]
Ssmal = hm.mat2list(S2)[(int)(L/2)]

valsS, vecsS = la.eigh(Ssmal)
Sdiag = vecsS.conj().T@Hsmal@vecsS
vals = np.round(valsS).astype('int')
counts = np.bincount(vals)
mask = (vals == np.argmax(counts))

Htiny = (vecsS.conj().T@Hsmal@vecsS)[mask][:,mask]
valsH, vecsH = la.eigh(Htiny)

np.save('data/level_stats_L' + str(L), valsH)
