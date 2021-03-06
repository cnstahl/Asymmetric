import numpy as np
import scipy.linalg as la
import hamiltonian as hm
#import matplotlib.pyplot as plt
import os.path

L     = 8
dense = True
#vs = np.asarray([5])
vs    = np.asarray([5,  6,  7,  8,  9, 10, 11, 12, 14, 16, 18, 20, 22, 24])
sites = np.arange(L)
pert_strength = 4

if (dense): 
    H = hm.dense_H(L)
    prefix = 'data/otoc_dense'
else: 
    H = hm.sparse_H(L)
    prefix = 'data/otoc_sparse'
if (not pert_strength == 0): 
    prefix = prefix + "_pert_"
H = H + hm.init_pert(L, pert_strength)
H = H + hm.finl_pert(L, pert_strength)
Hlist = hm.mat2list(H)
vals_list = []
vecs_list = []
vecsd_list = []
for idx, H in enumerate(Hlist):
    vals, vecs = la.eigh(H)
    vals_list.append(vals)
    vecs_list.append(vecs)
    vecsd_list.append(vecs.T.conj())

mask = np.zeros(len(vs), dtype=bool)
for idx, v in enumerate(vs):
    fname = prefix + "foreL" + str(L) + "v" + str(v) + ".npy"
    mask[idx] = not (os.path.isfile(fname))
vs = vs[mask]

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
    
weightsfore = []
weightsback = []

for idx, t in enumerate(times):
    (fore, _) = hm.get_weights_from_time_sites(L, t, sites_at_ts_fore[idx], vals_list, vecs_list, vecsd_list)
    (_, back) = hm.get_weights_from_time_sites(L, t, sites_at_ts_back[idx], vals_list, vecs_list, vecsd_list)
    weightsfore.append(fore)
    weightsback.append(back)
    
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
        otocsback[idx, site] = weightsback[i][j]

    np.save(prefix + "foreL" + str(L) + "v" + str(v), otocsfore[idx])
    np.save(prefix + "backL" + str(L) + "v" + str(v), otocsback[idx])
        
#ax = plt.subplot(111)
#for idx, otocfore in enumerate(otocsfore):
#    ax.plot(sites[::2], otocfore[::2], label = str(vs[idx]))
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
#ax.legend(bbox_to_anchor=(1.05, 1), loc=2)
#plt.ylim(0,1)
#plt.savefig('data/broadotocsfore_L_' + str(L))
#plt.close()
#    
#ax = plt.subplot(111)
#for idx, otocback in enumerate(otocsback):
#    ax.plot(sites[::2], otocback[::2], label = str(vs[idx]))
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
#ax.legend(bbox_to_anchor=(1.05, 1), loc=2)
#plt.ylim(0,1)
#plt.savefig('data/broadotocsback_L_' + str(L))
