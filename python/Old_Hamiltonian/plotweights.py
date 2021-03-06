import numpy as np
import matplotlib.pyplot as plt
from sys import argv

name = argv[1]

weights = np.load(name)

for weightlist in weights:
    L = len(weightlist)
    ax = plt.subplot(111)
    for i in range(L):
        ax.plot(np.arange(len(weightlist[i])),weightlist[i], label = str(i))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.ylim(0,1)
    plt.show()
