import numpy as np

def near(a, b, rtol = 1e-5, atol = 1e-8):
    return np.abs(a-b)<(atol+rtol*np.abs(b))

base = np.array([[1,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,1,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,1,0,1,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,1,0,1,1,0],
                 [0,0,0,0,0,0,0,1]])
eye  = np.eye(2, dtype='int')

def get_permutation(n):
    N = 2**n
    perm = np.zeros([N, N], dtype='int')
    for i in range(int(N/2)):
        perm[2*i, i] = 1
    for i in range(int(N/2)):
        perm[2*i + 1, i + int(N/2)] = 1
    return perm

def get_single_transition(n):
    if n < 3: 
        raise NameError('n is too small')
    if n == 3: 
        return base* 1
    else: 
        return np.kron(get_single_transition(n-1), eye)

# Transition matrix for gates on all sites with equal probability
def get_full_transition(n):
    single = get_single_transition(n)
    full   = single
    perm   = get_permutation(n)
    for i in range(n-1):
        single = perm.T @ single @ perm
        full += single
    return full

def get_reducer(n):
    if n == 3: return np.array([[1,0,0,0,0,0,0,0],
                                [0,1,0,0,0,0,0,0],
                                [0,0,1,0,0,0,0,0],
                                [0,0,0,0,1,0,0,0],
                                [0,0,0,1,0,0,0,0],
                                [0,0,0,0,0,1,0,0],
                                [0,0,0,0,0,0,1,0],
                                [0,0,0,0,0,0,0,1]])
    
    if n == 4: return np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                                [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                                [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])

def popcount_zero(x):
    c = 0
    while x:
        x &= x - 1
        c += 1

    return c

# mask to get one subset of transition matrix
def get_mask(n, rise):
    mask = np.zeros([2**n, 2**n], dtype='bool')
    ones = n + rise
    if (ones % 2 == 0): ones = int(ones / 2)
    else: raise NameError('invalid rise for number of sites')
        
    for i, row in enumerate(mask):
        if popcount_zero(i) == ones:
            for j, val in enumerate(row):
                if popcount_zero(j) == ones:
                    mask[i,j] = True
    return mask

def get_steady_states(n, rise):
    transition = get_full_transition(n)/n
    a = transition[get_mask(n, rise)]
    a = a.reshape((np.sqrt(len(a)), np.sqrt(len(a))))
    
    D, V = np.linalg.eig(a)
    V = V.T
    return V[near(D, 1.0)][0]

def digits(number, base=2):
    assert number >= 0
    if number == 0:
        return [0]
    l = []
    while number > 0:
        l.append(number % base)
        number = number // base
    return l

def get_correlation(val,n):
    a = np.array(digits(val))
#     print(a)
    extra = n-len(a)
    assert extra >= 0
    a = np.append(a, np.zeros(extra))
    a = 2*a-1
#     print(a)
    return (np.sum(a*np.roll(a,1))/len(a) - (np.average(a))**2) + 1/(n-1)

def get_cors(rise, n):
    mask = np.zeros(2**n, dtype='bool')
    ones = n + rise
    if (ones % 2 == 0): ones = int(ones / 2)
    else: raise NameError('invalid rise for number of sites')
        
    for i, row in enumerate(mask):
        if popcount_zero(i) == ones:
            mask[i] = True
    
    states = np.arange(2**n, dtype=float)[mask]
    
    for idx, val in enumerate(states):
        states[idx] = get_correlation(val,n)
#         print(get_correlation(val,n), states[idx])
    return states

rise = 0
for n in range(4,12,2):
    print("Correlation for length {}".format(n))
    state = get_steady_states(n, rise)
    state /= np.sum(state)
#    print(np.sum(state))
    cors = get_cors(rise, n)
#    print(np.average(cors))
    print(state@cors)
    print()
