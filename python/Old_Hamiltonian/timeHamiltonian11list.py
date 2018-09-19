import numpy as np
import scipy.linalg as  la
import hamiltonian as hm

L = 11

H = hm.sparse_H(L)
vals, vecs = la.eigh(H)
vecsd = vecs.T.conj()

# Total time elapsed
end = 1
# Time steps per second
n = 1
N = n*end
A = hm.Z
for i in range(L-1):
    A = np.kron(A,hm.I)
Alist = hm.mat2list(A)
B = hm.Z
for i in range(L-1):
    B = np.kron(hm.I,B)
Blist = hm.mat2list(B)


weightfore9 = np.empty((L, N))
weightback9 = np.empty((L, N))

for i in np.arange(N):
    t = i/n
#     unitt = np.matmul(np.matmul(vecs,  np.diag(np.exp(-1j*vals*t))), vecsd)
#     uninv = np.matmul(np.matmul(vecs,  np.diag(np.exp( 1j*vals*t))), vecsd)
    unitt = np.matmul(vecs * np.exp(-1j*vals*t), vecsd)
    uninv = np.matmul(vecs * np.exp( 1j*vals*t), vecsd)
    
    ulist = hm.mat2list(unitt)
    uinvlist = hm.mat2list(uninv)
    
#     At    = np.matmul(np.matmul(uninv, A),             unitt)
#     Bt    = np.matmul(np.matmul(uninv, B),             unitt)
#    At    = np.matmul(np.matmul(np.linalg.inv(unitt), A),             unitt)
#    Bt    = np.matmul(np.matmul(np.linalg.inv(unitt), B),             unitt)
#     print(chop(unit3t),"\n")
    Atlist = []
    for idx, val in enumerate(Alist):
        Atlist.append(np.matmul(np.matmul(uinvlist[idx], val), ulist[idx]))
    Btlist = []
    for idx, val in enumerate(Blist):
        Btlist.append(np.matmul(np.matmul(uinvlist[idx], val), ulist[idx]))
    At = hm.list2mat(Atlist)
    Bt = hm.list2mat(Btlist)
    
    front = hm.norm(At)
    back  = hm.norm(Bt)
    
    for j in range(L):
        At = hm.end_trace(At,1)
        Bt = hm.front_trace(Bt,1)
        fronthere = hm.norm(At)
        backhere  = hm.norm(Bt)
        weightfore9[L-1-j, i] = front - fronthere
        weightback9[j, i]     = back  - backhere
        front = fronthere
        back  = backhere
