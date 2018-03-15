import numpy as np
import scipy.linalg as  la
import hamiltonian as hm
from sys import argv

L = 13
# Total time elapsed
end = 1
# Time steps per second
n = 1
here = False
dense = False

weightfore, weightback = hm.get_all_weights(L, end, n, here=here, dense=dense)

fname = 'data/L' + str(L) + 'end' + str(end) + 'n' + str(n)
if (dense):
    fname = 'dense' + fname
if (here):
    fname = fname + '_here'
np.save(fname, [weightfore, weightback])
