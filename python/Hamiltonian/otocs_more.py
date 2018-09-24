import numpy as np
import scipy.linalg as la
import asymmetric as asym
import quantum as qm
import time
#import matplotlib.pyplot as plt
import glob

num_trials = 3
L     = 11
dense = True
# Get vs list from loading
sites = np.arange(L)
pert_strength = 2.5
h = 2
_, _, _, sig_z_list = qm.get_sigma_lists(L)

if (dense):
    fix = 'dense'
    H = asym.dense_H(L)
    prefix = 'data/otoc_dense'
else:
    fix = 'sparse'
    H = asym.sparse_H(L)
    prefix = 'data/otoc_sparse'
if (not pert_strength == 0):
    fix    = fix    + "_pert_"
    prefix = prefix + "_pert_"

# Load existing data

forenames = glob.glob(prefix + "foreL" + str(L) + "v*.npy")
backnames = glob.glob(prefix + "backL" + str(L) + "v*.npy")
otocsfore_all = []
otocsback_all = []
vs        = []
vsback    = []
for fname in forenames:
    otocsfore_all.append(np.load(fname))
    vs.append(    int(fname.replace(prefix + "foreL" + str(L) + "v", "").replace(".npy", "")))
for fname in backnames:
    otocsback_all.append(np.load(fname)[::-1])
    vsback.append(int(fname.replace(prefix + "backL" + str(L) + "v", "").replace(".npy", "")))
vs        = np.array(vs)
otocsfore_all = np.array(otocsfore_all)
otocsback_all = np.array(otocsback_all)
args      = np.argsort(vs)
vs        = vs[args]
otocsfore_all = otocsfore_all[args]
otocsback_all = otocsback_all[np.argsort(vsback)]

# Start making new stuff

H = H + asym.init_pert(L, pert_strength)
H0 = H + asym.finl_pert(L, pert_strength)

start = time.time()
for q in range(num_trials):
    H = H0 + qm.get_local_field(sig_z_list, np.random.rand(L)*2*h - h)
    Hlist = asym.mat2list(H)
    vals_list = []
    vecs_list = []
    vecsd_list = []
    for idx, H in enumerate(Hlist):
        vals, vecs = la.eigh(H)
        vals_list.append(vals)
        vecs_list.append(vecs)
        vecsd_list.append(vecs.T.conj())

    times = []
    for v in vs: times.extend(sites/v)
    times = list(dict.fromkeys(times))
    times.sort()
    # print(times)

    sites_at_ts_fore = []
    sites_at_ts_back = []
    for ttime in times:
        sites_at_t_fore = []
        sites_at_t_back = []
        for v in vs:
            dist = (int) (np.round(ttime*v))
            site_fore = dist
            site_back = L-dist-1
            if np.isclose((ttime*v), dist) and (site_fore < L) and not site_fore in sites_at_t_fore:
                sites_at_t_fore.append(site_fore)
            if np.isclose((ttime*v), dist) and (site_back >-1) and not site_back in sites_at_t_back:
                sites_at_t_back.append(site_back)
        sites_at_ts_fore.append(sites_at_t_fore)
        sites_at_ts_back.append(sites_at_t_back)

    weightsfore = []
    weightsback = []

    for idx, t in enumerate(times):
#        print(t)
        (fore, _) = asym.get_weights_from_time_sites(
                L, t, sites_at_ts_fore[idx], vals_list, vecs_list, vecsd_list)
        (_, back) = asym.get_weights_from_time_sites(
                L, t, sites_at_ts_back[idx], vals_list, vecs_list, vecsd_list)
        weightsfore.append(fore)
        weightsback.append(back)

    otocsfore = np.zeros((len(vs), 1, L))
    otocsback = np.zeros((len(vs), 1, L))
    for idx, v in enumerate(vs):
        for site in range(L):
            t_need = site/v
            for i, t in enumerate(times):
                if np.isclose(t,t_need): break
            j = (sites_at_ts_fore[i]).index(site)
    #         print(v, site, sites_at_ts_fore[i][j], t_need, times[i])
            otocsfore[idx, 0, site] = weightsfore[i][j]

        for dist in range(L):
            site = L-dist-1
            t_need = dist/v
            for i, t in enumerate(times):
                if np.isclose(t,t_need): break
            j = (sites_at_ts_back[i]).index(site)
    #         print(v, site, sites_at_ts_back[i][j], t_need, times[i])
            otocsback[idx, 0, site] = weightsback[i][j]

    # print(np.shape(otocsfore_all), np.shape(otocsfore))
    otocsfore_all = np.append(otocsfore_all, otocsfore, axis=1)
    otocsback_all = np.append(otocsback_all, otocsback, axis=1)

    for idx, v in enumerate(vs):
        np.save(prefix + "foreL" + str(L) + "v" + str(v), otocsfore_all[idx])
        np.save(prefix + "backL" + str(L) + "v" + str(v), otocsback_all[idx])

    end = time.time()
    print('trials:', q, 'seconds:', end-start)

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
