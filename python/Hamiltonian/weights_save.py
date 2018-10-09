import numpy as np
import asymmetric as asym
import glob

runs = 4
L = 9
end = 3
n = 20

here = True
pauli = True
Azero = True
Aplus = False
Amult = False
field_strength = 1
dot_strength = 0

prefix = "data/weights_L"+str(L)+"end"+str(end)+"n"+str(n)+"_"+str(int(here))+ \
         str(int(pauli))+"_"+str(int(Azero))+str(int(Aplus))+str(int(Amult))+ \
         "_f"+str(field_strength)+"d"+str(dot_strength)+"#"

for _ in range(runs):
    # Get data
    weights = asym.get_all_weights(L, end, n, here, pauli, True,
                        dot_strength, field_strength,
                        Azero, Aplus, Amult)

    # Save data
    existing = glob.glob(prefix + "*.npy")
    highest=-1
    for fname in existing:
        current = int(fname.replace(prefix, "").replace(".npy", ""))
        if current>highest: highest=current
    np.save(prefix+str(highest+1), weights)
