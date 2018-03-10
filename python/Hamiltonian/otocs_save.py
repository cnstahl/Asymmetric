import numpy as np
import scipy.linalg as la
import hamiltonian as hm
import matplotlib.pyplot as plt

L     = 11
vs    = np.asarray([8, 10, 12, 14, 16, 18, 20, 22, 24])
sites = np.arange(L)

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
    
H = hm.dense_H(L)
vals, vecs = la.eigh(H)

weightsfore = []
weightsback = []

for idx, t in enumerate(times):
    (fore, _) = hm.get_weights(L, t, sites_at_ts_fore[idx], vecs=vecs, vals=vals)
    (_, back) = hm.get_weights(L, t, sites_at_ts_back[idx], vecs=vecs, vals=vals)
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
for idx, v in enumerate(vs):
    for dist in range(L):
        site = L-dist-1
        t_need = dist/v
        for i, t in enumerate(times):
            if np.isclose(t,t_need): break
        j = (sites_at_ts_back[i]).index(site)
#         print(v, site, sites_at_ts_back[i][j], t_need, times[i])
        otocsback[idx, site] = weightsback[i][j]

for idx, v in enumerate(vs):
    np.save("data/otocforeL" + str(L) + "v" + str(v), otocsfore[idx])
    np.save("data/otocbackL" + str(L) + "v" + str(v), otocsback[idx])
        
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
