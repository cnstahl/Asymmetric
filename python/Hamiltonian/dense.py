import asymmetric as asym
import numpy as np

L = 11
# Total time elapsed
end = 3
# Time steps per second
n = 20

weightfore9, weightback9 = asym.get_all_weights(L, end, n, here=False, dense = True)
np.save('data/denseweights', [weightfore9, weightback9])
