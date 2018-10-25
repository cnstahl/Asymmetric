import numpy as np
import scipy.linalg as la
import asymmetric as asym
import quantum as qm
#import matplotlib.pyplot as plt
import os.path
import glob

L     = 14
dense = True
field_strength = 1
nvecs = 2
cutoff = 0
# vs    = np.asarray([1, 3, 5,  6,  7,  8,  9, 10, 11, 12, 14, 16, 18, 20, 22, 24])
vs    = np.asarray([1, 2, 3, 4, 6, 8, 12, 16, 18])
#vs    = np.asarray([1, 3, 5, 7, 9, 12, 15, 18, 20])
sites = np.arange(L)

prefix = 'data/zotoc_vbL' + str(L) + "#"
times = []
for v in vs: times.extend(sites/v)
times = list(dict.fromkeys(times))
times.sort()
# print(times)

sites_at_ts_fore = []
sites_at_ts_back = []
for time in times:
    sites_at_t_fore = []
    sites_at_t_back = []
    for v in vs:
        dist = (int) (np.round(time*v))
        site_fore = dist
        site_back = L-dist-1
        if np.isclose((time*v), dist) and (site_fore < L) and not site_fore in sites_at_t_fore:
            sites_at_t_fore.append(site_fore)
        if np.isclose((time*v), dist) and (site_back >-1) and not site_back in sites_at_t_back:
            sites_at_t_back.append(site_back)
    sites_at_ts_fore.append(sites_at_t_fore)
    sites_at_ts_back.append(sites_at_t_back)

H = asym.dense_H(L)
_,x_list,y_list, z_list = qm.get_sigma_lists(L, half=False)
if (not field_strength is None):
    h = field_strength/2 # Take into account spin-1/2
    H = H + qm.get_local_field(z_list, np.random.rand(L)*2*h - h)

Hlist  = asym.mat2list(H)
Zlists = [asym.mat2list(Z) for Z in z_list]
# s_Hlist  =  [H for H in Hlist  if H.shape[0]<cutoff]
# s_Zlists = [[Z for Z in z_list if Z.shape[0]<cutoff] for z_list in Zlists]
# s_weightsfore = []
# s_weightsback = []
#
# for idx, t in enumerate(times):
#     fore = asym.zotoc_ed_sites(Hlist, Zlists, sites_at_ts_fore[idx], t, fore=True)
#     back = asym.zotoc_ed_sites(Hlist, Zlists, sites_at_ts_back[idx], t, fore=False)
#     s_weightsfore.append(fore)
#     s_weightsback.append(back)

l_Hlist  =  [H for H in Hlist  if H.shape[0]>=cutoff]
l_Zlists = [[Z for Z in z_list if Z.shape[0]>=cutoff] for z_list in Zlists]
l_weightsfore = []
l_weightsback = []

vec = qm.get_vec_Haar(2**L)
vecs = asym.arr2list(vec)
l_vecs = [v for v in vecs   if len(v)>=cutoff]
for idx, t in enumerate(times):
    fore = asym.zotoc_vec_sites(Hlist, vecs, Zlists, sites_at_ts_fore[idx], t, fore=True)
    back = asym.zotoc_vec_sites(Hlist, vecs, Zlists, sites_at_ts_back[idx], t, fore=False)
    l_weightsfore.append(fore)
    l_weightsback.append(back)
for n in range(nvecs-1):
    vec = qm.get_vec_Haar(2**L)
    vecs = asym.arr2list(vec)
    l_vecs = [v for v in vecs   if len(v)>=cutoff]
    for idx, t in enumerate(times):
        fore = asym.zotoc_vec_sites(Hlist, vecs, Zlists, sites_at_ts_fore[idx], t, fore=True)
        back = asym.zotoc_vec_sites(Hlist, vecs, Zlists, sites_at_ts_back[idx], t, fore=False)
        l_weightsfore[idx] += fore
        l_weightsback[idx] += back

# weightsfore = [s + l/nvecs for (s,l) in zip(s_weightsfore, l_weightsfore)]
# weightsback = [s + l/nvecs for (s,l) in zip(s_weightsback, l_weightsback)]
weightsfore = [l/nvecs for l in l_weightsfore]
weightsback = [l/nvecs for l in l_weightsback]

####### Post processing #######

otocsfore = np.zeros((len(vs), L))
otocsback = np.zeros((len(vs), L))
for idx, v in enumerate(vs):
    for site in range(L):
        t_need = site/v
        for i, t in enumerate(times):
            if np.isclose(t,t_need): break
        j = (sites_at_ts_fore[i]).index(site)
#         print(v, site, sites_at_ts_fore[i][j], t_need, times[i])
        otocsfore[idx, site] = weightsfore[i][j]

    for dist in range(L):
        site = L-dist-1
        t_need = dist/v
        for i, t in enumerate(times):
            if np.isclose(t,t_need): break
        j = (sites_at_ts_back[i]).index(site)
#         print(v, site, sites_at_ts_back[i][j], t_need, times[i])
        otocsback[idx, dist] = weightsback[i][j]

# Save data
existing = glob.glob(prefix + "*.npy")
highest=-1
for fname in existing:
    current = int(fname.replace(prefix, "").replace(".npy", ""))
    if current>highest: highest=current
np.save(prefix+str(highest+1), np.stack((otocsfore, otocsback)))
