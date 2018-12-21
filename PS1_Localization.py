import numpy as np

""" 
Monte Carlo localization: Histogram filter
discrete states; multi-modal
"""

# Assume that at each step, the robot:
# 1) first makes a movement,
# 2) then takes a measurement.


def localize(world, measurements, motions, sensor_right, p_move):
    """
    :param world: ndarray
    Map of the world
    :param measurements:
    :param motions:
    :param sensor_right: the probability that any given measurement is correct
    :param p_move: probability that any given movement command takes place
    :return:
    p: the probabilities that the robot occupies each cell in the world
    """
    p_init = 1 / world.size     # assuming uniform probability
    p = np.ones(world.shape) * p_init
    for u, z in zip(motions, measurements):
        p = move(p, u, p_move)
        p = sense(world, p, z, sensor_right)

    return p


def sense(world, pri, z, sensor_right):
    q = np.zeros(pri.shape)    # posterior
    for i, j in np.ndindex(world.shape):
        if world[i, j] == z:
            q[i, j] = pri[i, j] * sensor_right
        else:
            q[i, j] = pri[i, j] * (1 - sensor_right)
    # normalization
    q = q / np.sum(q)
    return q


def move(pri, u, p_move):
    q = np.zeros(pri.shape)
    dy = u[0]
    dx = u[1]
    for i, j in np.ndindex(pri.shape):
        row_ex = (i + dy) % pri.shape[0]        # move exactly
        col_ex = (j + dx) % pri.shape[1]

        q[row_ex, col_ex] += pri[i, j] * p_move
        q[i, j] += pri[i, j] * (1 - p_move)       # cases that it fails to move

    return q


# initialization
world = [['R', 'G', 'G', 'R', 'R'],
         ['R', 'R', 'G', 'R', 'R'],
         ['R', 'R', 'G', 'G', 'R'],
         ['R', 'R', 'R', 'R', 'R']]
world = np.array(world)

measurements = ['G', 'G', 'G', 'G', 'G']
motions = [[0, 0], [0, 1], [1, 0], [1, 0], [0, 1]]
sensor_right = 0.7
p_move = 0.8

p = localize(world, measurements, motions, sensor_right, p_move)
print(p)
