import numpy as np
import scipy.linalg as la
import asymmetric as asym
import quantum as qm
#import matplotlib.pyplot as plt
import os.path

L     = 9
dense = True
field_strength = 1
# vs    = np.asarray([1, 3, 5,  6,  7,  8,  9, 10, 11, 12, 14, 16, 18, 20, 22, 24])
vs    = np.asarray([1, 2, 3, 4, 6, 8, 12, 16, 18])
#vs    = np.asarray([1, 3, 5, 7, 9, 12, 15, 18, 20])
bonds = np.arange(L-1)+.5

prefix = 'data/zotoc_dense'
H = asym.dense_H(L)
_,x_list,y_list, z_list = qm.get_sigma_lists(L, half=False)
if (not field_strength is None):
    h = field_strength/2 # Take into account spin-1/2
    H = H + qm.get_local_field(z_list, np.random.rand(L)*2*h - h)

Hlist  = asym.mat2list(H)
Zlists = [asym.mat2list(Z) for Z in z_list]
vec = qm.get_vec_Haar(2**L)
vecs = asym.arr2list(vec)

mask = np.zeros(len(vs), dtype=bool)
for idx, v in enumerate(vs):
    fname = prefix + "foreL" + str(L) + "v" + str(v) + ".npy"
    mask[idx] = not (os.path.isfile(fname))
vs = vs[mask]

times = []
for v in vs: times.extend(bonds/v)
times = list(dict.fromkeys(times))
times.sort()
# print(times)

sites_at_ts_fore = []
sites_at_ts_back = []
for time in times:
    sites_at_t_fore = set([])
    sites_at_t_back = set([])
    for v in vs:
        dist = np.round(time*v)
        if np.isclose(dist%1, .5):
            site_fore_0 = (int) (  dist-0.5)
            site_fore_1 = (int) (  dist+0.5)
            site_back_0 = (int) (L-dist-0.5)
            site_back_1 = (int) (L-dist-1.5)
            if (site_fore_0 < L-1): sites_at_t_fore.update(site_fore_0, site_fore_1)
            if (site_back_0 <   0): sites_at_t_back.update(site_back_0, site_back_1)
    sites_at_ts_fore.append(list(sites_at_t_fore))
    sites_at_ts_back.append(list(sites_at_t_back))

weightsfore = []
weightsback = []

for idx, t in enumerate(times):
    fore = asym.zotoc_vec_sites(Hlist, vecs, Zlists, sites_at_ts_fore[idx], t, fore=True)
    back = asym.zotoc_vec_sites(Hlist, vecs, Zlists, sites_at_ts_back[idx], t, fore=False)
    weightsfore.append(fore)
    weightsback.append(back)

otocsfore = np.zeros((len(vs), L-1))
otocsback = np.zeros((len(vs), L-1))
for idx, v in enumerate(vs):
    for bond in np.arange(L-1)+.5:
        t_need = bond/v
        for i, t in enumerate(times):
            if np.isclose(t,t_need): break
        j = (sites_at_ts_fore[i]).index((int) (bond-.5))
        k = (sites_at_ts_fore[i]).index((int) (bond+.5))
        otocsfore[idx, bond] = (weightsfore[i][j] + weightsfore[i][k])/2

    for dist in np.arange(L-1)+.5:
        bond = L-dist-1
        t_need = dist/v
        for i, t in enumerate(times):
            if np.isclose(t,t_need): break
        j = (sites_at_ts_back[i]).index((int) (bond-.5))
        k = (sites_at_ts_back[i]).index((int) (bond+.5))
        otocsback[idx, bond] = (weightsback[i][j] + weightsback[i][k])/2

    np.save(prefix + "foreL" + str(L) + "v" + str(v), [otocsfore[idx]])
    np.save(prefix + "backL" + str(L) + "v" + str(v), [otocsback[idx]])
