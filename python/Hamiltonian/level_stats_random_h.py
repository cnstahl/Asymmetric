import numpy as np
import quantum as qm
import asymmetric as asym
import scipy.sparse as sparse
import scipy.linalg as la

levels = 25
Ls = [8, 10, 12, 14]
data = np.zeros((len(Ls),2,levels))
for i, L in enumerate(Ls):

    trials = 2600 - 200*L
    if (trials < 10): trials = 100
    rs = np.zeros((trials, levels))
    hs = np.logspace(-1,2, levels)

    # Only create these once per L
    H0 = asym.dense_H(L)
    _, _, _, sig_z_list = ([sig/2 for sig in sigs] for sigs in qm.get_sigma_lists(L))

    for idx, h in enumerate(hs):
        for j in range(trials):
            H_pert = qm.get_local_field(sig_z_list, np.random.rand(L)*2*h - h)/2
            H = H0 + H_pert
            choose = L//2
            rs[j, idx] = qm.get_r(asym.mat2list(H)[choose].A, nonz=True)

    data[i] = qm.mean_and_std(rs)

np.save("data/phasetrans", data)
