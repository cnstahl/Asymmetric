import numpy as np
import quantum as qm
import asymmetric as asym
import scipy.sparse.linalg as spla
import glob

runs = 1
L = 11
end = 20
n = 3
tot = end*n
field_strength = 1

prefix = "data/zotocs_L"+str(L)+"end"+str(end)+"n"+str(n)+"_"+ \
          str(field_strength)+"#"

_,x_list,y_list, z_list = qm.get_sigma_lists(L, half=False)
H = asym.dense_H(L)
Z0 = z_list[0]
if (not field_strength is None):
    h = field_strength/2 # Take into account spin-1/2
    H = H + qm.get_local_field(z_list, np.random.rand(L)*2*h - h)

Hlist  = asym.mat2list(H)
Zlists = [asym.mat2list(Z) for Z in z_list]
Z0list = Zlists[0]

vec = qm.get_vec_Haar(2**L)
vecs = asym.arr2list(vec)

e = spla.expm_multiply
OTOCs = np.zeros((L,tot))
for T in range(tot):
    t = T/n
    vbs  = [e(1j*H*t, Z0@e(-1j*H*t, vec)) for (H, Z0, vec) in zip(Hlist, Z0list, vecs)]

    for i in range(L):
        v1s = [e(1j*H*t, Z0@e(-1j*H*t, Zi@vec)) for (H, Z0, vec, Zi) in zip(Hlist, Z0list, vecs, Zlists[i])]
        v2s = [Zi@vb for (Zi, vb) in zip(Zlists[i], vbs)]
        OTOCs[i, T] = 1-sum([v2.conj().T@v1 for (v1, v2) in zip(v1s, v2s)]).real

# Save data
existing = glob.glob(prefix + "*.npy")
highest=-1
for fname in existing:
    current = int(fname.replace(prefix, "").replace(".npy", ""))
    if current>highest: highest=current
np.save(prefix+str(highest+1), OTOCs)
