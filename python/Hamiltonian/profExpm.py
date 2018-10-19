import numpy as np
import scipy.linalg as lapack

L=8
H = asym.dense_H(L)
U1 = la.expm(-1j*H)

Hlist = asym.mat2list(H)
U2list = list(map(la.expm, H))

vals_list = []
vecs_list = []
vecsd_list = []
for idx, H in enumerate(Hlist):
    vals, vecs = la.eigh(H)
    vals_list.append(vals)
    vecs_list.append(vecs)
#     vecsd_list.append(vecs.T.conj())
