DEPRECATED!! See notebook
import numpy as np
import matplotlib.pyplot as plt
import quantum as qm
import asymmetric as asym

L = 11
trials = 20
levels = 10
rs = np.zeros((trials, levels))
hs = np.linspace(1,10, levels)

# Only create these once
H0 = asym.dense_H(L)
_, _, _, sig_z_list = qm.get_sigma_lists(L)

for idx, h in enumerate(hs):
    for j in range(trials):
        H_pert = qm.get_local_field(sig_z_list, np.random.rand(L)*2*h - h)
        H = H0 + H_pert
        rs[j, idx] = qm.get_r(asym.mat2list(H)[L//2], avg=True)

mean, std = qm.mean_and_std(rs)
plt.errorbar(
    hs,
    mean,
    yerr = std,
    marker = '.',
    drawstyle = 'steps-mid-'
)
plt.xlabel('$h$', fontsize=15)
plt.ylabel('$r$', fontsize=15)
plt.ylim(.4,.6)
plt.title("Level repulsion" + ", L=" + str(L), fontsize=15)
plt.savefig("figures/levelrepul" + "L" + str(L) + "small_.pdf")
