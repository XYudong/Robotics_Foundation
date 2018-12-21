import numpy as np

p = [0.2, 0.2, 0.2, 0.2, 0.2]       # probability of each spot on the map(i.e. a probability distribution)
world = ['green', 'red', 'red', 'green', 'green']       # map
Z = ['red', 'red']        # measurements
motions = [1, 1]

# for measurements
pHit = 0.6
pMiss = 0.2
# for motions
pExact = 0.8
pOvershoot = 0.1
pUndershoot = 0.1


def sense(pri, z):
    """update the input belief through production"""
    q = np.zeros(len(p))    # posterior
    for i in range(len(world)):
        if world[i] == z:
            q[i] = pri[i] * pHit
        else:
            q[i] = pri[i] * pMiss
    # normalization
    q = q / np.sum(q)
    return q


def move(pri, u):
    """shift the prior belief with uncertainty through convolution"""
    num_spots = len(pri)
    p_post = np.zeros(num_spots)
    for i in range(num_spots):
        idx_exact = (i + u) % num_spots
        idx_under = (i + u - 1) % num_spots
        idx_over = (i + u + 1) % num_spots

        p_post[idx_exact] += pExact * pri[i]
        p_post[idx_under] += pUndershoot * pri[i]
        p_post[idx_over] += pOvershoot * pri[i]

    return p_post


for z, u in zip(Z, motions):
    p_post = sense(p, z)
    p = move(p_post, u)

print(p)

# ff = lambda x: - x * np.log(x)





