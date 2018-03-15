import numpy as np
import scipy.linalg as  la
import hamiltonian as hm

L = 9

H = hm.sparse_H(L)
Hlist = hm.mat2list(H)
vals_list = []
vecs_list = []
vecsd_list = []
for idx, H in enumerate(Hlist):
    vals, vecs = la.eigh(H)
    vals_list.append(vals)
    vecs_list.append(vecs)
    vecsd_list.append(vecs.T.conj())

# Total time elapsed
end = 3
# Time steps per second
n = 20
N = n*end
A = np.array([hm.Z[0,0], hm.Z[1,1]])
for i in range(L-1):
    A = np.kron(A,np.array([1,1]))
Alist = hm.arr2list(A)
B = np.array([hm.Z[0,0], hm.Z[1,1]])
for i in range(L-1):
    B = np.kron(np.array([1,1]),B)
Blist = hm.arr2list(B)


weightfore9 = np.empty((L, N))
weightback9 = np.empty((L, N))

for i in np.arange(N):
    t = i/n
    ulist = []
    uinvlist = []
    for idx, vecs in enumerate(vecs_list):
        ulist.append(   np.matmul(vecs * np.exp(-1j*vals_list[idx]*t), vecsd_list[idx]))
        uinvlist.append(np.matmul(vecs * np.exp( 1j*vals_list[idx]*t), vecsd_list[idx]))
    
    Atlist = []
    for idx, val in enumerate(Alist):
        Atlist.append(np.matmul(uinvlist[idx] * val, ulist[idx]))
    Btlist = []
    for idx, val in enumerate(Blist):
        Btlist.append(np.matmul(uinvlist[idx] * val, ulist[idx]))
    At = hm.list2mat(Atlist)
    Bt = hm.list2mat(Btlist)
    
    front = 1
    back  = 1
    
    for j in range(L):
        At = hm.end_trace(At,1)
        Bt = hm.front_trace(Bt,1)
        fronthere = hm.norm(At)
        backhere  = hm.norm(Bt)
        weightfore9[L-1-j, i] = front - fronthere
        weightback9[j, i]     = back  - backhere
        front = fronthere
        back  = backhere

np.save('9site.weights', [weightfore9,weightback9])
