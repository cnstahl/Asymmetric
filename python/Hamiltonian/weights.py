import numpy as np
import scipy.linalg as  la
import sparsehamiltonian as hm
from sys import argv

L = 13
# Total time elapsed
end = 3
# Time steps per second
n = 20
here = False
dense = False

weightfore, weightback = hm.get_all_weights(L, end, n, here=here, dense=dense)

fname = 'data/L' + str(L) + 'end' + str(end) + 'n' + str(n)
if (dense):
    fname = 'dense' + fname
if (here):
    fname = fname + '_here'
np.save(fname, [weightfore, weightback])
