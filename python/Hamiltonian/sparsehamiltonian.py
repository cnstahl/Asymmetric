import numpy as np
import scipy.sparse.linalg as  la
import scipy.linalg as npla
import scipy.special
import scipy.sparse as sp

I = sp.csc_matrix([[ 1,  0],[ 0,  1]])
X = sp.csc_matrix([[ 0,  1],[ 1,  0]])
Y = sp.csc_matrix([[ 0,-1j],[1j,  0]])
Z = sp.csc_matrix([[ 1,  0],[ 0, -1]])
H3mult = sp.csc_matrix([[0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 , 0],
               [0 ,  0 ,  1 ,  0 , -1 ,  0 ,  0 , 0],
               [0 , -1 ,  0 ,  0 ,  1 ,  0 ,  0 , 0],
               [0 ,  0 ,  0 ,  0 ,  0 , -1 ,  1 , 0],
               [0 ,  1 , -1 ,  0 ,  0 ,  0 ,  0 , 0],
               [0 ,  0 ,  0 ,  1 ,  0 ,  0 , -1 , 0],
               [0 ,  0 ,  0 , -1 ,  0 ,  1 ,  0 , 0],
               [0 ,  0 ,  0 ,  0 ,  0 ,  0 ,  0 , 0]])

def sparse_Hmult(l):
    if (l==3): return H3mult
#     print(np.shape(np.kron(sparse_Hmult(l-2),I2)))
#     print(np.shape(np.kron(np.eye(l-3), H3mult)))
    return (sp.kron(sparse_Hmult(l-2), sp.csc_matrix(np.eye(4))) + 
            sp.kron(np.eye(2**(l-3)), H3mult))

def sparse_H(l):
    if (l%2 != 1): assert False, "l must be odd"
    H = sparse_Hmult(l)
    return H * (2j * np.pi)/(3 * np.sqrt(3))

def dense_Hmult(l):
    if (l==3): return H3mult
#     print(np.shape(np.kron(sparse_Hmult(l-2),I2)))
#     print(np.shape(np.kron(np.eye(l-3), H3mult)))
    return (sp.kron(dense_Hmult(l-1), sp.csc_matrix(np.eye(2))) + 
            sp.kron(np.eye(2**(l-3)), H3mult))

def dense_H(l):
    if (l < 3): assert False, "l must be >=3"
    H = dense_Hmult(l)
    return H * (2j * np.pi)/(3 * np.sqrt(3))

def chop(a):
    if not np.all(np.isclose(np.imag(a),0)): 
        print("\nchop() removed the imaginary part\n ")
    A = np.round(np.real(a),3)
    if np.all(np.isclose(A.astype(int), A)): return A.astype(int)
    else: return A

def norm(A):
    norm = np.trace(np.matmul(A, A.T.conj()))/(A.shape[0])
#    norm = (A.dot(A.T.conj())).diagonal().sum()/(len(A))
#    print("here")
    assert np.isclose(np.imag(norm),0)
    return np.real(norm)

# Trace out the last/first i spins
def end_trace(x, i):
    N = len(x)
    untraced = (int) (N/2**i)
    traced = 2**i
    x = x.reshape((untraced, traced, untraced, traced))
    return np.trace(x, axis1=1, axis2=3)/2**i

def front_trace(x, i):
    N = len(x)
    untraced = (int) (N/2**i)
    traced = 2**i
    x = x.reshape((traced, untraced, traced, untraced))
    return np.trace(x, axis1=0, axis2=2)/2**i

def tensor(*args):
    A = 1
    for arg in args:
        A = np.kron(A, arg)
    return A

def par_tr(x,i):
    N = len(x)
    assert 2**i < N
    
    indices = np.array(range(N))
    bit = int((N/2)/2**i)
    mask = N - (bit) - 1
    indices = indices & mask
    return (x[:,np.unique(indices)][np.unique(indices)] + 
            x[:,np.unique(indices + bit)][np.unique(indices + bit)])/2
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
    assert (len(np.shape(matrix)) == 2)
    L = (int) (np.log2(matrix.shape[0]))
    alph2Sz, Sz2alph = permutations(L)
    diag = matrix[alph2Sz]
    diag = diag[:,alph2Sz]
    A = []
    j = 0
    for i in range(L+1):
        k = j + (int) (scipy.special.comb(L,i))
        A.append(diag[j:k, j:k])
        j = k
    return A

def list2mat(A):
    L = len(A) - 1
    alph2Sz, Sz2alph = permutations(L)
#    diag = sp.block_diag((*A), format='csc')
    diag = npla.block_diag(*A)
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
def get_weights_from_time_sites(L, t, sites, vals_list, vecs_list, vecsd_list, here=True):

    # Get preliminary stuff
    A = np.array([Z[0,0], Z[1,1]])
    for i in range(L-1):
        A = np.kron(A,np.array([1,1]))
    Alist = arr2list(A)
    print("Made Alist", flush=True)
    B = np.array([Z[0,0], Z[1,1]])
    for i in range(L-1):
        B = np.kron(np.array([1,1]),B)
    Blist = arr2list(B)
    print("Made Alist, Blist", flush=True)

    weightfore = np.empty(len(sites))
    weightback = np.empty(len(sites))
    
    ulist = []
    uinvlist = []
    for idx, vecs in enumerate(vecs_list):
        ulist.append(   np.matmul(vecs * np.exp(-1j*vals_list[idx]*t), vecsd_list[idx]))
        uinvlist.append(np.matmul(vecs * np.exp( 1j*vals_list[idx]*t), vecsd_list[idx]))
    print("Made ulist, et ", flush=True)
    Atlist = []
    for idx, val in enumerate(Alist):
        Atlist.append(np.matmul(uinvlist[idx] * val, ulist[idx]))
    Btlist = []
    for idx, val in enumerate(Blist):
        Btlist.append(np.matmul(uinvlist[idx] * val, ulist[idx]))
    At = list2mat(Atlist)
    Bt = list2mat(Btlist)
    print("Evolved At, Bt", flush=True)
    
    front = 1
    back  = 1
    
    if (here):
        for j, site in enumerate(sites):
            Aj = par_tr(At,site)
            Bj = par_tr(Bt,site)
            fronthere = norm(Aj)
            backhere  = norm(Bj)
            weightfore[j] = 1 - fronthere
            weightback[j] = 1 - backhere
    elif (not here):
        for j in range(L):
            At = end_trace(At,1)
            Bt = front_trace(Bt,1)
            fronthere = norm(At)
            backhere  = norm(Bt)
            weightfore[L-1-j] = front - fronthere
            weightback[j]     = back  - backhere
            front = fronthere
            back  = backhere
    else: assert False, "Should never get here"
    print("Finished weights", flush=True)
    return np.array([weightfore, weightback])

# Get (L x N) matrix containing all weights
def get_all_weights(L, end, n, here=True, dense = False):
    if (dense): H = dense_H(L)
    else: H = sparse_H(L)
    print("Made H")
    Hlist = mat2list(H)
    vals_list = []
    vecs_list = []
    vecsd_list = []
    for idx, H in enumerate(Hlist):
        if (H.shape[0] == 1):
            vals = H.todense()
            vecs = np.array([[1]])
        else:
            vals, vecs = la.eigsh(H, k=H.shape[0]-2)
#        vals, vecs = npla.eigh(H.todense())
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
