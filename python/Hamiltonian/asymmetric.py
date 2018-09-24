import quantum as qm
import scipy.special
import scipy.sparse as sparse
import scipy.linalg as  la
import numpy as np

ident = qm.ident

H3mult = sparse.csr_matrix([[0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 , 0],
               [0 ,  0 ,  1 ,  0 , -1 ,  0 ,  0 , 0],
               [0 , -1 ,  0 ,  0 ,  1 ,  0 ,  0 , 0],
               [0 ,  0 ,  0 ,  0 ,  0 , -1 ,  1 , 0],
               [0 ,  1 , -1 ,  0 ,  0 ,  0 ,  0 , 0],
               [0 ,  0 ,  0 ,  1 ,  0 ,  0 , -1 , 0],
               [0 ,  0 ,  0 , -1 ,  0 ,  1 ,  0 , 0],
               [0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 , 0]])

def sparse_Hmult(l):
    if (l==3): return H3mult
    return (sparse.kron(sparse_Hmult(l-2), ident(4), format='csr') +
            sparse.kron(ident(2**(l-3)), H3mult, format='csr'))

def sparse_H(l):
    if (l%2 != 1): assert False, "l must be odd"
    return sparse_Hmult(l) * 1j

def dense_Hmult(l):
    if (l==3): return H3mult
#     print(np.shape(np.kron(sparse_Hmult(l-2),I2)))
#     print(np.shape(np.kron(np.eye(l-3), H3mult)))
    return (sparse.kron(dense_Hmult(l-1), ident(2), format='csr') +
            sparse.kron(ident(2**(l-3)), H3mult, format='csr'))

def dense_H(l):
    if (l < 3): assert False, "l must be >=3"
    return dense_Hmult(l) * 1j


def init_pert(L, pert_strength):
    pert = sparse.csr_matrix([[ 0,  0,  0,  0],
                 [ 0, -1,  1,  0],
                 [ 0,  1, -1,  0],
                 [ 0,  0,  0,  0]])
    for i in range(L-2):
        pert = sparse.kron(pert, ident(2))
    return pert*pert_strength

def finl_pert(L, pert_strength):
    pert = sparse.csr_matrix([[ 0,  0,  0,  0],
                 [ 0, -1,  1,  0],
                 [ 0,  1, -1,  0],
                 [ 0,  0,  0,  0]])
    for i in range(L-2):
        pert = sparse.kron(ident(2), pert)
    return pert*pert_strength


def permutations(L):
    alph2Sz = np.zeros(2**L, dtype=int)
    for i in range(2**L):
        alph2Sz[i] = bin(i).count('1')
    alph2Sz = alph2Sz.argsort()
    Sz2alph = np.zeros(2**L, dtype=int)
    for idx, val in enumerate(alph2Sz):
        Sz2alph[val] = idx
    return alph2Sz, Sz2alph

def mat2list(matrix):
    L = (int) (np.log2(matrix.shape[0]))
    alph2Sz, Sz2alph = permutations(L)
    diag = matrix[alph2Sz]
    diag = diag[:,alph2Sz]
    A = []
    j = 0
    for i in range(L+1):
        k = j + (int) (scipy.special.comb(L,i))
        A.append(diag[j:k, j:k].A)
        j = k
    return A

def list2mat(A):
    L = len(A) - 1
    alph2Sz, Sz2alph = permutations(L)
    A = list(map(sparse.csr_matrix, A))
    diag = sparse.block_diag(A, format='csr')
    mat = diag[Sz2alph]
    return mat[:,Sz2alph]

def arr2list(array):
    assert (len(np.shape(array)) == 1)
    L = (int) (np.log2(len(array)))
    alph2Sz, Sz2alph = permutations(L)
    perm = array[alph2Sz]
    A = []
    j = 0
    for i in range(L+1):
        k = j + (int) (scipy.special.comb(L,i))
        A.append(perm[j:k])
        j = k
    return A

# def list2arr(A):
#     L = len(A) - 1
#     alph2Sz, Sz2alph = permutations(L)
#     diag = la.block_diag(*A)
#     mat = diag[Sz2alph]
#    return mat[:,Sz2alph]

# Get weights at only some sites at a given time
# Pass (vals, vecs) for faster performance
def get_weights_from_time_sites(L, t, sites, vals_list, vecs_list,
                                vecsd_list, here=True):

    # Get preliminary stuff
    A = np.array([1, -1])
    for i in range(L-1):
        A = np.kron(A,np.array([1,1]))
    Alist = arr2list(A)
    B = np.array([1, -1])
    for i in range(L-1):
        B = np.kron(np.array([1,1]),B)
    Blist = arr2list(B)


    weightfore = np.empty(len(sites))
    weightback = np.empty(len(sites))

    ulist = []
    uinvlist = []
    for idx, vecs in enumerate(vecs_list):
        ulist.append(   np.matmul(vecs * np.exp(-1j*vals_list[idx]*t),
                                  vecsd_list[idx]))
        uinvlist.append(np.matmul(vecs * np.exp( 1j*vals_list[idx]*t),
                                  vecsd_list[idx]))

    Atlist = []
    for idx, val in enumerate(Alist):
        Atlist.append(np.matmul(uinvlist[idx] * val, ulist[idx]))
    Btlist = []
    for idx, val in enumerate(Blist):
        Btlist.append(np.matmul(uinvlist[idx] * val, ulist[idx]))
    At = list2mat(Atlist)
    Bt = list2mat(Btlist)

    front = 1
    back  = 1

    if (here):
        for j, site in enumerate(sites):
            Aj = qm.par_tr(At,site)
            Bj = qm.par_tr(Bt,site)
            fronthere = qm.mat_norm(Aj)
            backhere  = qm.mat_norm(Bj)
            weightfore[j] = 1 - fronthere
            weightback[j] = 1 - backhere
    elif (not here):
        for j in range(L):
            At = qm.end_trace(At,1)
            Bt = qm.front_trace(Bt,1)
            fronthere = qm.mat_norm(At)
            backhere  = qm.mat_norm(Bt)
            weightfore[L-1-j] = front - fronthere
            weightback[j]     = back  - backhere
            front = fronthere
            back  = backhere
    else: assert False, "Should never get here"

    return np.array([weightfore, weightback])

# Get (L x N) matrix containing all weights
def get_all_weights(L, end, n, here=True, dense = False, pert_strength=0,
                    finl_pert_strength=None, ising_strength=None):
    if (dense): H = dense_H(L)
    else: H = sparse_H(L)
    if (finl_pert_strength == None): finl_pert_strength = pert_strength
    if (not pert_strength==0):
        H = H + init_pert(L, pert_strength)
        H = H + finl_pert(L, finl_pert_strength)
    if (not ising_strength==None): H = H + ising_strength * ising_mult(L)
    Hlist = mat2list(H)
    vals_list = []
    vecs_list = []
    vecsd_list = []
    for idx, H in enumerate(Hlist):
        vals, vecs = la.eigh(H)
        vals_list.append(vals)
        vecs_list.append(vecs)
        vecsd_list.append(vecs.T.conj())

    N = n*end

    weightfore = np.empty((L, N))
    weightback = np.empty((L, N))

    for i in np.arange(N):
        t = i/n
        weightfore[:,i], weightback[:,i] = \
                          get_weights_from_time_sites(L, t, range(L), vals_list, vecs_list, vecsd_list, here=here)

    return weightfore, weightback
