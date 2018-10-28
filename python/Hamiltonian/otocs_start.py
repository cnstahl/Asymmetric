import numpy as np
import scipy.linalg as la
import asymmetric as asym
import quantum as qm
#import matplotlib.pyplot as plt
import os.path

L     = 12
dense = True
#vs = np.asarray([5])
# vs    = np.asarray([1, 3, 5,  6,  7,  8,  9, 10, 11, 12, 14, 16, 18, 20, 22, 24])
vs    = np.asarray([1, 2, 3, 4, 6, 8, 12, 16, 18])
#vs    = np.asarray([1, 3, 5, 7, 9, 12, 15, 18, 20])
sites = np.arange(L)
pert_strength = 0
h = .1
_, _, _, sig_z_list = qm.get_sigma_lists(L)

if (dense):
    H = asym.dense_H(L)
    prefix = 'data/otoc_dense'
else:
    H = asym.sparse_H(L)
    prefix = 'data/otoc_sparse'
if (not pert_strength == 0):
    prefix = prefix + "_pert_"
prefix = prefix + "h" + str(h)
H = H + asym.init_pert(L, pert_strength)
H = H + asym.finl_pert(L, pert_strength)
H = H + qm.get_local_field(sig_z_list, np.random.rand(L)*2*h - h)
Hlist = asym.mat2list(H)
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
    (fore, _) = asym.get_weights_from_time_sites(L, t, sites_at_ts_fore[idx], vals_list, vecs_list, vecsd_list)
    (_, back) = asym.get_weights_from_time_sites(L, t, sites_at_ts_back[idx], vals_list, vecs_list, vecsd_list)
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

    np.save(prefix + "foreL" + str(L) + "v" + str(v), [otocsfore[idx]])
    np.save(prefix + "backL" + str(L) + "v" + str(v), [otocsback[idx]])

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
