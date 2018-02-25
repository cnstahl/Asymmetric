import hamiltonian as hm
import numpy as np

L = 11
H = hm.sparse_H(L)

alph2Sz = np.zeros(2**L, dtype=int)
for i in range(2**L):
    alph2Sz[i] = bin(i).count('1')
alph2Sz = alph2Sz.argsort()
Sz2alph = np.zeros(2**L, dtype=int)
for idx, val in enumerate(alph2Sz):
    Sz2alph[val] = idx

# plt.matshow(np.imag(H))
# plt.show()
Hdiag = H[alph2Sz]
Hdiag = Hdiag[:,alph2Sz]
# plt.matshow(np.imag(Hdiag))
# plt.show()

H2 = Hdiag[Sz2alph]
H2 = H2[:,Sz2alph]
# print(np.all(np.isclose(H,H2)))
