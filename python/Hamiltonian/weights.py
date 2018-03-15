import numpy as np
import scipy.linalg as  la
import hamiltonian as hm
from sys import argv

L = 11
# Total time elapsed
end = 3
# Time steps per second
n = 20

weightfore, weightback = hm.get_all_weights(L, end, n, here=False, dense = False)

np.save("data/" + argv[0].replace(".py", ""), [weightfore, weightback])
