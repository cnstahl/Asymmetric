import scipy.sparse as sparse
import scipy.sparse.linalg as spalin
import numpy as np
import itertools
from scipy import linalg
import os
import sys
from itertools import chain, combinations
import scipy
import time
import operator
import functools
import copy

# Takes in a binary number b stored as an array.
def bin2dec(b):
    b= np.array(b)[::-1]
    d=0
    for i in range(b.shape[0]):
        d=d+b[i]*2**i
    return d

def dec2bin(d,n):
    a =  [0]*(n - len(list(bin(d)[2:]))) + list(bin(d)[2:])
    return np.array(a,dtype=int)

def powerset(iterable):
    xs = list(iterable)
    # note we return an iterator rather than a list
    return list(chain.from_iterable( combinations(xs,n) for n in range(len(xs)+1) ))

def revbits(x,L):
    return int(bin(x)[2:].zfill(L)[::-1], 2)

def flipbits(x,L):
    dim=2**L
    return dim-x-1


def gen_reflection(L):
    reflectMat = sparse.lil_matrix((2**L, 2**L))
    for i in range(2**L):
        reflectMat[revbits(i,L),i]=1
    return sparse.csr_matrix(reflectMat)




def diagonalizeWithSymmetries(H, PList,L):
    ind = 0
    Allevecs = np.zeros((2**L, 2**L), 'complex')
    Allevals = np.zeros((2**L))
    for P in PList:
        H_sym = P*H*(P.T)
        if (H_sym - np.conj(H_sym)).size ==0:
            H_sym = np.real(np.array(H_sym.todense()))
        else:
            H_sym = np.array(H_sym.todense())

        evals, evecs= linalg.eigh(H_sym)
        Allevecs[:, ind:ind+len(evecs)] = np.dot(np.array((P.T).todense()), evecs)
        Allevals[ind:ind+len(evecs)] = evals
        ind = ind+len(evecs)
    ind = np.argsort(Allevals)
    Allevals=Allevals[ind]
    Allevecs=Allevecs[:,ind]

    return Allevals, Allevecs


def EigenvaluesWithSymmetries(H, PList,L):
    ind = 0
    Allevals = np.zeros((2**L))
    for P in PList:
        H_sym = P*H*(P.T)
        if (H_sym - np.conj(H_sym)).size ==0:
            H_sym = np.real(np.array(H_sym.todense()))
        else:
            H_sym = np.array(H_sym.todense())

        evals= linalg.eigvalsh(H_sym)
        Allevals[ind:ind+len(evals)] = evals
        ind = ind+len(evals)
    Allevals= np.sort(Allevals)

    return Allevals



def gen_s0sxsysz(L): 
    sx = sparse.csr_matrix([[0., 1.],[1., 0.]]) 
    sy = sparse.csr_matrix([[0.,-1j],[1j,0.]]) 
    sz = sparse.csr_matrix([[1., 0],[0, -1.]])
    s0_list =[]
    sx_list = [] 
    sy_list = [] 
    sz_list = []
    I = sparse.csr_matrix(np.eye(2**L))
    for i_site in range(L):
        if i_site==0: 
            X=sx 
            Y=sy 
            Z=sz 
        else: 
            X= sparse.csr_matrix(np.eye(2)) 
            Y= sparse.csr_matrix(np.eye(2)) 
            Z= sparse.csr_matrix(np.eye(2))
            
        for j_site in range(1,L): 
            if j_site==i_site: 
                X=sparse.kron(X,sx, 'csr')
                Y=sparse.kron(Y,sy, 'csr') 
                Z=sparse.kron(Z,sz, 'csr') 
            else: 
                X=sparse.kron(X,np.eye(2),'csr') 
                Y=sparse.kron(Y,np.eye(2),'csr') 
                Z=sparse.kron(Z,np.eye(2),'csr')
        sx_list.append(X)
        sy_list.append(Y) 
        sz_list.append(Z)
        s0_list.append(I)

    return s0_list, sx_list,sy_list,sz_list 


# efficient way to generate the diagonal Sz lists without doing sparse matrix multiplications
def gen_sz(L):
    sz_list = []
    for i in range(L-1,-1,-1):
        unit =[1]*(2**i)+[-1]*(2**i)
        sz_list.append(np.array(unit*(2**(L-i-1))))
    return sz_list


def gen_op_total(op_list):
    L = len(op_list)
    tot = op_list[0]
    for i in range(1,L): 
        tot = tot + op_list[i] 
    return tot

def gen_op_prod(op_list):
    L= len(op_list)
    P = op_list[0]
    for i in range(1, L):
        P = P*op_list[i]
    return P

def gen_onsite_field(op_list, h_list):
    L= len(op_list)
    H = h_list[0]*op_list[0]
    for i in range(1,L): 
        H = H + h_list[i]*op_list[i] 
    return H

       
# generates \sum_i O_i O_{i+k} type interactions
def gen_interaction_kdist(op_list, op_list2=[],k=1, J_list=[], bc='obc'):
    L= len(op_list)

    if op_list2 ==[]:
        op_list2=op_list
    H = sparse.csr_matrix(op_list[0].shape)
    if J_list == []:
        J_list =[1]*L
    Lmax = L if bc == 'pbc' else L-k
    print(Lmax)
    for i in range(Lmax):
        H = H+ J_list[i]*op_list[i]*op_list2[np.mod(i+k,L)]
    return H        
    
def gen_nn_int(op_list, J_list=[], bc='obc'):
    return gen_interaction_kdist(op_list,op_list, 1, J_list, bc)

    
def gen_lr_int(op_list, alpha, bc='obc'):
    L= len(op_list)
    interaction=0*op_list[0]
    for i in range(L):
        for j in range(i+1,L):
            r=1.0*abs(i-j)
            if bc=='pbc':
                r=np.min([r, L-r])
            interaction = interaction+ (op_list[i]*op_list[j])/(r**alpha)
    return interaction
 
            

def gen_diag_projector(symMatrix, symValue):
    symMatrix = symMatrix.diagonal()
    ind = np.where(symMatrix==symValue)
    dim = len(symMatrix)
    dim0 = np.size(ind)
    P = sparse.lil_matrix((dim0,dim ))
    for i in range(np.size(ind)):
        P[i,ind[0][i]] = 1.0
    return P


def projectedStateNum_ToState(symMatrix, symValue, psnum):
    symMatrixDense = symMatrix.todense()
    ind = np.where(np.diag(symMatrixDense)==symValue)
    dimOrig = len(symMatrixDense)
    
    return dec2bin(dimOrig - 1 -ind[0][psnum], int(np.log2(dimOrig)))
    
    
def gen_state_bloch(thetaList, phiList):
    L=len(thetaList)
    psi = np.kron([np.cos(thetaList[0]/2.),np.exp(1j*phiList[0])*np.sin(thetaList[0]/2.)],
                  [np.cos(thetaList[1]/2.),np.exp(1j*phiList[1])*np.sin(thetaList[1]/2.)])
    for i in range(2,L):
        psi = np.kron(psi, [np.cos(thetaList[i]/2.),np.exp(1j*phiList[i])*np.sin(thetaList[i]/2.)])
    return psi


def gen_unifstate_bloch(theta, phi,L):
    return gen_state_bloch([theta]*L, [phi]*L)


def LevelStatistics(energySpec, nbins = 25, ret=True):
    delta = energySpec[1:] -energySpec[0:-1]
    delta = abs(delta[abs(delta) > 10**-12])
    r = map(lambda x,y: min(x,y)*1.0/max(x,y), delta[1:], delta[0:-1])
    if ret==True:
        return np.array(r), np.mean(r)
    return np.mean(r)