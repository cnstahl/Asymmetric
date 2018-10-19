import numpy as np
import scipy.sparse as sparse
import scipy.linalg as  la
###import scipy.special

################################################################################
##                                                                            ##
## Useful bits                                                                ##
################################################################################

def ident(len):
    return sparse.csr_matrix(np.eye(len))

def chop(a, warn = True):
    if sparse.issparse(a): a = a.toarray()
    if not np.all(np.isclose(np.imag(a),0)):
        if warn: print("\nchop() removed the imaginary part\n ")
    A = np.round(np.real(a),3)
    if np.all(np.isclose(A.astype(int), A)): return A.astype(int)
    else: return A

def tensor(*args):
    A = 1
    for arg in args:
        A = np.kron(A, arg)
    return A

def mat_norm(A):
    assert sparse.issparse(A)
    norm = np.trace((A*A.T.conj()).toarray())/A.shape[0]
    assert np.isclose(np.imag(norm),0)
    return np.real(norm)

# get r value (average ratio between adjacent gaps) from a matrix
def get_r(mat, avg=True, frac=1/3, nonz=False):
    vals, _ = la.eigh(mat)
    # print(vals)
    if nonz:
        vals = vals[np.isclose(np.isclose(vals,0),0)]
        vals = vals[vals<0]
        size = (int) (len(vals)*frac*2)
        vals = vals[range(len(vals)-size, len(vals))]
    else: print(not supported)
    # print(vals)
    delta = (vals - np.roll(vals, 1))[1:]
    stats = (np.minimum(delta, np.roll(delta, 1)) / np.maximum(delta, np.roll(delta, 1)))[1:]
    if avg: return np.average(stats)
    return stats

# Take an array of data, and return a tuple containing mean and std
# Average over the 0th dimension
def mean_and_std(data):
    trials = len(data)
    mean   = np.mean(data, axis=0)
    std    = np.std(data, axis=0)/np.sqrt(trials-1)
    return mean, std

################################################################################
##                                                                            ##
## Sigma Matrices                                                             ##
################################################################################

I      = sparse.csr_matrix([[ 1,  0],[ 0,  1]])
sig_x  = sparse.csr_matrix([[0., 1.],[1., 0.]])
sig_y  = sparse.csr_matrix([[ 0,-1j],[1j,  0]])
sig_z  = sparse.csr_matrix([[ 1,  0],[ 0, -1]])

def get_sigma_lists(L):
    sig_0_list = []
    sig_x_list = []
    sig_y_list = []
    sig_z_list = []
    sig0 = ident(2**L)
    for site_i in range(L):
        if site_i == 0:
            X = sig_x
            Y = sig_y
            Z = sig_z
        else:
            X = I
            Y = I
            Z = I
        for site_j in range(1,L):
            if site_j == site_i:
                X = sparse.kron(X, sig_x, 'csr')
                Y = sparse.kron(Y, sig_y, 'csr')
                Z = sparse.kron(Z, sig_z, 'csr')
            else:
                X = sparse.kron(X, I,     'csr')
                Y = sparse.kron(Y, I,     'csr')
                Z = sparse.kron(Z, I,     'csr')
        sig_0_list.append(sig0)
        sig_x_list.append(X)
        sig_y_list.append(Y)
        sig_z_list.append(Z)
    return sig_0_list, sig_x_list, sig_y_list, sig_z_list

################################################################################
##                                                                            ##
## Density Matrix Operations                                                  ##
################################################################################

# Trace out the last/first i spins
def end_trace(x, i):
    assert sparse.issparse(x)
    N = np.log2(x.shape[0])
    for site in range(i):
        x = par_tr(x, N-1-site)
    return x

def front_trace(x, i):
    assert sparse.issparse(x)
    if i==1:
        return par_tr(x, 0)
    else:
        return par_tr(front_trace(x, i-1), 0)

def par_tr(x,i):
    assert sparse.issparse(x)
    N = x.shape[0]
    assert 2**i < N

    indices = np.array(range(N))
    bit = int((N/2)/2**i)
    mask = N - (bit) - 1
    indices = indices & mask
    return (x[:,np.unique(indices)][np.unique(indices)] +
            x[:,np.unique(indices + bit)][np.unique(indices + bit)])/2

################################################################################
##                                                                            ##
## Operator Operations                                                        ##
################################################################################

def op_sum(list):
    return sum(list)
    # L = len(list)
    # S = list[0]
    # for i in range(1,L):
    #     S += list[i]
    # return S

def op_prod(list):
    L = len(list)
    P = list[0]
    for i in range(1,L):
        P *= list[i]
    return P

def get_local_field(op_list, h_list, warn=True):
    if warn:
        if not len(op_list) == len(h_list): print("truncating field")
    return sum([op*h for op,h in zip(op_list, h_list)])
