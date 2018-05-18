import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import seaborn as sns
#sns.set()
#sns.set_style("white")
#sns.set_palette(sns.hls_palette(11, l=.5, s=1))
colors = ["aqua", "dark blue"]
sns.set_palette(sns.color_palette("Set1", 9)+sns.xkcd_palette(colors))

name = argv[1]

weights = np.load(name)
n = 20
k = 0
names = ["OTOC for Forward Wave", "OTOC for Backward Wave"]
names = ["Pauli Weight for Forward Wave", "Pauli Weight for Backward Wave"]
filenames = ["front", "back"]

for weightlist in weights:
    L = len(weightlist)
    ax = plt.subplot(111)
    for i in range(L):
        ax.plot(np.arange(len(weightlist[i]))/n,weightlist[i], label = str(i))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.ylim(0,1)
    plt.ylabel("$W(i,t)$", fontsize=18)
    plt.xlabel("$t$", fontsize=18)
#    print(i)
#    print(names)
    plt.title(names[k], fontsize=18)
    plt.savefig("figures/" + name.replace(".npy", "").replace("data/", "") + filenames[k] + ".pdf")
    plt.close()
    k += 1
