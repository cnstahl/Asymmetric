import numpy as np
import quantum as qm
import asymmetric as asym
import scipy.sparse.linalg as spla
import importlib
from scipy.special import  comb

def zotoc_vec_std(Hlist, vecs, Zlists, sites, t):
    e = spla.expm_multiply
    L = len(sites)
    Z0list = Zlists[0]
    vbs  = [e(1j*H*t, Z0@e(-1j*H*t, vec)) for (H, Z0, vec) in zip(Hlist, Z0list, vecs)]

    OTOCs = np.zeros((L, L+1))
    for idx, site in enumerate(sites):
        v1s = [e(1j*H*t, Z0@e(-1j*H*t, Zi@vec)) for (H, Z0, vec, Zi) in zip(Hlist, Z0list, vecs, Zlists[site])]
        v2s = [Zi@vb for (Zi, vb) in zip(Zlists[site], vbs)]
        OTOCs[idx] = [(v2.conj().T@v1).real for (v1, v2) in zip(v1s, v2s)]
    return OTOCs

L = 10
_,_,_, z_list = qm.get_sigma_lists(L, half=False)
H = asym.dense_H(L)
h = .1
H = H + qm.get_local_field(z_list, np.random.rand(L)*2*h - h)

Hlist  = asym.mat2list(H)
Zlists = [asym.mat2list(Z) for Z in z_list]

nvecs = 5
sites = np.arange(L)
t = 20
OTOCss = np.zeros((nvecs,L,L+1))
for i in range(nvecs):
    vec = qm.get_vec_Haar(2**L)
    vecs = asym.arr2list(vec)
    OTOCss[i] = zotoc_vec_std(Hlist, vecs, Zlists, sites, t)

np.save('data/vec_size_OTOCs', OTOCss)
