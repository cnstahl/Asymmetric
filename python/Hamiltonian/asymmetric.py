import quantum as qm
import scipy.special
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
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
    return sparse_Hmult(l) * 1j / 8

def dense_Hmult(l):
    if (l==3): return H3mult
#     print(np.shape(np.kron(sparse_Hmult(l-2),I2)))
#     print(np.shape(np.kron(np.eye(l-3), H3mult)))
    return (sparse.kron(dense_Hmult(l-1), ident(2), format='csr') +
            sparse.kron(ident(2**(l-3)), H3mult, format='csr'))

def dense_H(l):
    if (l < 3): assert False, "l must be >=3"
    return dense_Hmult(l) * 1j / 8


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
        A.append(diag[j:k, j:k])
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

# Get zotocs, exactly, using full Hilbert space
def zotoc_ed_sites(Hlist, Zlists, sites, t, fore=True):
    Z0list = Zlists[0] if fore else Zlists[-1]
    Ulist    = [spla.expm(-1j*H*t) for H in Hlist]
    Ulistinv = [spla.expm( 1j*H*t) for H in Hlist]
    Z0tlist  = [Ui@Z0@U for (Ui, Z0, U) in zip(Ulistinv, Z0list, Ulist)]

    OTOCs = np.zeros(len(sites))
    for idx, site in enumerate(sites):
        corr = [Z0t@Zi@Z0t@Zi for (Z0t, Zi) in zip(Z0tlist, Zlists[site])]
        OTOCs[idx] = 1-sum([c.diagonal().sum().real for c in corr])/2**len(Zlists)
    return OTOCs

def zotoc_mat_exact(L, Hlist, Zlists, end=20, n=3, fore=True):
    tot = end*n
    sites = range(L)

    OTOCs = np.zeros((L,tot))
    for T in range(tot):
        t = T/n
        OTOCs[:, T] = zotoc_ed_sites(Hlist, Zlists, sites, t, fore)
    return OTOCs

# Get zotocs, using expm_multiply, projecting onto a vector
def zotoc_vec_sites(Hlist, vecs, Zlists, sites, t, fore=True):
    e = spla.expm_multiply
    Z0list = Zlists[0] if fore else Zlists[-1]
    vbs  = [e(1j*H*t, Z0@e(-1j*H*t, vec)) for (H, Z0, vec) in zip(Hlist, Z0list, vecs)]

    OTOCs = np.zeros(len(sites))
    for idx, site in enumerate(sites):
        v1s = [e(1j*H*t, Z0@e(-1j*H*t, Zi@vec)) for (H, Z0, vec, Zi) in zip(Hlist, Z0list, vecs, Zlists[site])]
        v2s = [Zi@vb for (Zi, vb) in zip(Zlists[site], vbs)]
        OTOCs[idx] = 1-sum([v2.conj().T@v1 for (v1, v2) in zip(v1s, v2s)]).real
    return OTOCs

def zotoc_vec_expm(L, Hlist, vecs, Zlists, end=20, n=3, fore=True):
    tot = end*n
    sites = range(L)

    OTOCs = np.zeros((L,tot))
    for T in range(tot):
        t = T/n
        OTOCs[:, T] = zotoc_vec_sites(Hlist, vecs, Zlists, sites, t, fore)
    return OTOCs

# Use hybrid methods for blocks
def zotoc_hy_sites(Hlist, vecs, Zlists, sites, t, fore=True):
    not implemented
    s_Hlist  =  [H for H in Hlist  if H.shape[0]<cutoff]
    s_Zlists = [[Z for Z in z_list if Z.shape[0]<cutoff] for z_list in Zlists]

    l_Hlist  =  [H for H in Hlist  if H.shape[0]>=cutoff]
    l_Zlists = [[Z for Z in z_list if Z.shape[0]>=cutoff] for z_list in Zlists]
    l_vecs  =   [v for v in vecs   if len(v)>=cutoff]

    return zotoc_ed_sites(s_Hlist, s_Zlists, sites, t, fore)

# Get weights at some sites at a given time
# Do here and/or pauli, and any inits we might want
def get_weights_from_time_sites(L, t, sites, vals_list, vecs_list, vecsd_list,
                                here=True, pauli=False, As=[]):
    # Size of return array
    num_weights = (Azero+Aplus+Amult) * (pauli+here) * 2 # For front and back
    # ret = np.zeros((num_weights, L))
    ret = []
    weightfore = np.zeros(len(sites))
    weightback = np.zeros(len(sites))

    # Get evolution matrices
    ulist = []
    uinvlist = []
    for idx, vecs in enumerate(vecs_list):
        ulist.append(   np.matmul(vecs * np.exp(-1j*vals_list[idx]*t),
                                  vecsd_list[idx]))
        uinvlist.append(np.matmul(vecs * np.exp( 1j*vals_list[idx]*t),
                                  vecsd_list[idx]))

    # For each requested init, evolve it forward then get weightsback
    for idx, (A,B) in enumerate(As):
        # Evolve Forward
        Alist = arr2list(A)
        Atlist = []
        for idx, val in enumerate(Alist):
            Atlist.append(np.matmul(uinvlist[idx] * val, ulist[idx]))
        At = list2mat(Atlist)

        Blist = arr2list(B)
        Btlist = []
        for idx, val in enumerate(Blist):
            Btlist.append(np.matmul(uinvlist[idx] * val, ulist[idx]))
        Bt = list2mat(Btlist)

        # Get weights
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
            ret.append(weightfore.copy())
            ret.append(weightback.copy())
        if (pauli):
            for j in range(L):
                At = qm.end_trace(At,1)
                Bt = qm.front_trace(Bt,1)
                fronthere = qm.mat_norm(At)
                backhere  = qm.mat_norm(Bt)
                weightfore[L-1-j] = front - fronthere
                weightback[j]     = back  - backhere
                front = fronthere
                back  = backhere
            ret.append(weightfore.copy())
            ret.append(weightback.copy())

    return np.array(ret)

def get_vecs_vals(L, dense=True, dot_strength=None, field_strength=None):
    # Construct Hamiltonian
    if (dense): H = dense_H(L)
    else: H = sparse_H(L)
    if (not dot_strength==None):
        H = H + init_pert(L, dot_strength)
        H = H + finl_pert(L, dot_strength)
    if (not field_strength==None):
        _, _, _, sig_z_list = ([sig/2 for sig in sigs] for sigs in
                               qm.get_sigma_lists(L))
        h = field_strength
        H = H + qm.get_local_field(sig_z_list, np.random.rand(L)*2*h - h)
    Hlist = mat2list(H)
    # Diagonalize
    vals_list = []
    vecs_list = []
    vecsd_list = []
    for idx, H in enumerate(Hlist):
        vals, vecs = la.eigh(H)
        vals_list.append(vals)
        vecs_list.append(vecs)
        vecsd_list.append(vecs.T.conj())

    return vals_list, vecs_list, vecsd_list

# Get all available data
def get_all_weights(L, end, n, here=True, pauli=None, dense=True,
                    dot_strength=None, field_strength=None,
                    Azero=True, Aplus=False, Amult=False):
    if (pauli==None): pauli = not here
    num_weights = (Azero+Aplus+Amult) * (pauli+here) * 2 # For front and back
    N = n*end
    ret = np.zeros((num_weights, L, N))

    # Get vecs and vals
    vals_list, vecs_list, vecsd_list = get_vecs_vals(L, dense, dot_strength,
                                                     field_strength)

    # Make init matrices
    As = []
    if (Azero):
        A = np.array([1, -1])
        for i in range(L-1):
            A = np.kron(A,np.array([1,1]))
        B = np.array([1, -1])
        for i in range(L-1):
            B = np.kron(np.array([1,1]),B)
        As.append((A,B))
    if (Aplus):
        A = np.array([1, 0, 0, -1])*np.sqrt(2)
        for i in range(L-2):
            A = np.kron(A,np.array([1,1]))
        B = np.array([1, 0, 0, -1])*np.sqrt(2)
        for i in range(L-2):
            B = np.kron(np.array([1,1]),B)
        As.append((A,B))
    if (Amult):
        A = np.array([1, -1, -1, 1])
        for i in range(L-2):
            A = np.kron(A,np.array([1,1]))
        B = np.array([1, -1, -1, 1])
        for i in range(L-2):
            B = np.kron(np.array([1,1]),B)
        As.append((A,B))

    # Get all the weights we want at each time
    for i in np.arange(N):
        t = i/n
        ret[:,:,i] = get_weights_from_time_sites(L, t, range(L), vals_list,
                                                 vecs_list, vecsd_list,
                                                 here,pauli, As)

    return ret
