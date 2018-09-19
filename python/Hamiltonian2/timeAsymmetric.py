import asymmetric as asym

L = 11
# Total time elapsed
end = 3
# Time steps per second
n = 3

weightfore9, weightback9 = asym.get_all_weights(L, end, n, here=False, dense = True)
