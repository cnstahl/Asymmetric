import numpy as np
import scipy.linalg as  la
import hamiltonian as hm
from sys import argv

L = 13

print('started')
H = hm.sparse_H(L)
print('Built Hamiltonian')
vals, vecs = la.eigh(H)
print('got eigensystem')
vecsd = vecs.T.conj()

# Total time elapsed
end = 1
# Time steps per second
n = 1
N = n*end
A = np.array([hm.Z[0,0], hm.Z[1,1]])
print('Built A')
for i in range(L-1):
    A = np.kron(A,np.array([1,1]))
Alist = hm.arr2list(A)
print('Built Alist')
B = np.array([hm.Z[0,0], hm.Z[1,1]])
print('Built B')
for i in range(L-1):
    B = np.kron(np.array([1,1]),B)
print('Built Blist')
Blist = hm.arr2list(B)


weightfore = np.empty((L, N))
weightback = np.empty((L, N))

for i in np.arange(N):
    t = i/n
    unitt = np.matmul(vecs * np.exp(-1j*vals*t), vecsd)
    uninv = np.matmul(vecs * np.exp( 1j*vals*t), vecsd)
    print('Built unitt, unitv')

    ulist = hm.mat2list(unitt)
    uinvlist = hm.mat2list(uninv)
    print('Built unitlists')
    
    Atlist = []
    for idx, val in enumerate(Alist):
        Atlist.append(np.matmul(uinvlist[idx] * val, ulist[idx]))
    Btlist = []
    for idx, val in enumerate(Blist):
        Btlist.append(np.matmul(uinvlist[idx] * val, ulist[idx]))
    At = hm.list2mat(Atlist)
    Bt = hm.list2mat(Btlist)
    print('Evolved A and B')
    
    front = 1
    back  = 1
    
    for j in range(L):
        At = hm.end_trace(At,1)
        Bt = hm.front_trace(Bt,1)
        fronthere = hm.norm(At)
        backhere  = hm.norm(Bt)
        weightfore[L-1-j, i] = front - fronthere
        weightback[j, i]     = back  - backhere
        front = fronthere
        back  = backhere
        print('Finished time', i)

np.save("data/" + argv[0].replace(".py", ""), [weightfore, weightback])
