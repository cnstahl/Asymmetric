import numpy as np
import matplotlib.pyplot as plt
from sys import argv

name = argv[1]

weights = np.load(name)

for weightlist in weights:
    L = len(weightlist)
    for i in range(L):
        plt.plot(np.arange(len(weightlist[i])),weightlist[i], label = str(i))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.ylim(0,1)
    plt.show()
