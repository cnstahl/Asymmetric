import sys
import os
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools
import operator
import functools
import matplotlib as mpl

import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)

from SpinLanguageSparse import*



def tripleprod(i, x,y,z, bc='obc'):
    L = len(x)
    if bc == 'obc' and i> L-3:
        return
    else:
        Det = x[i]*(y[np.mod(i+1,L)]*z[np.mod(i+2,L)]- y[np.mod(i+2,L)]*z[np.mod(i+1,L)])\
              -y[i]*(x[np.mod(i+1,L)]*z[np.mod(i+2,L)]- x[np.mod(i+2,L)]*z[np.mod(i+1,L)])\
              +z[i]*(x[np.mod(i+1,L)]*y[np.mod(i+2,L)]- x[np.mod(i+2,L)]*y[np.mod(i+1,L)])

        return Det
            


for L in range(3, 13):
    tt=time.time()
    s0, x, y, z = gen_s0sxsysz(L)


    Inv = gen_reflection(L)
    Pz = gen_op_prod(z)

    gmean = 0
    gvar = 0.5

    g1 = np.random.uniform(gmean-gvar,gmean+gvar,L/2)
    g2 = -g1[::-1]
    gcenter = list(np.random.uniform(0,0,1))*np.mod(L,2)
    g = np.append(np.append(g1,gcenter) ,g2)
    print g

    H = tripleprod(0,x,y,z)
    for i in range(1, L-2):
        H=H+tripleprod(i,x,y,z)

    H = H + gen_onsite_field(z, g)

    ## Check anticommutation and commutation
    print abs(H*Inv+Inv*H).max(), abs(H*Pz-Pz*H).max()


    symOp = gen_op_total(z).diagonal()
    PList = []
    vals = np.unique(symOp)
    for v in vals:
        ind = np.where(symOp==v)
        dim = np.size(ind)
        P = sparse.lil_matrix((dim,2**L))
        for j in range(dim):
            P[j,ind[0][j]] = 1.0

        PList.append(P)

    evals, evecs = diagonalizeWithSymmetries(H, PList, L)

    evals[abs(evals)<10**-12]=0
    zeroinds = np.where(abs(evals)<10**-12)[0]
    print L, len(zeroinds)

    if len(zeroinds)>0:
        zeroEvecs = evecs[:,zeroinds]
        InvMat = np.dot(np.dot((np.conj(zeroEvecs.T)), Inv.toarray()), zeroEvecs)
        ee, ev = linalg.eigh(InvMat)
        print ee, np.sum(ee)
    


    print ""






