import numpy as np
import time

"""Implement a ParticleFilter where each particle is a Robot"""


class Robot:
    # world
    world_size = 100        # assume a square world
    landmarks = [[0.0, 100.0], [0.0, 0.0], [100.0, 0.0], [100.0, 100.0]]  # position of 4 landmarks in (y, x) format.

    # robot
    max_steering_angle = np.pi / 4

    def __init__(self, length=10):
        """the robot is modeled as bicycle model"""
        # states
        self.x = np.random.random() * Robot.world_size
        self.y = np.random.random() * Robot.world_size
        self.orientation = np.random.random() * (2 * np.pi)     # [0, 2*pi]
        # other parameters
        self.length = length        # vehicle length
        self.bearing_noise = 0      # for measurements
        self.steering_noise = 0     # for movements
        self.distance_noise = 0

    def set(self, x, y, ori):
        self.x = x
        self.y = y
        self.orientation = ori
        return

    def set_noise(self, b_noise, s_noise, d_noise):
        self.bearing_noise = b_noise
        self.steering_noise = s_noise
        self.distance_noise = d_noise
        return

    def move(self, mv_cmd):
        """
        :param mv_cmd: [steering angle, distance passed by rear wheel]
        :return: a new Robot object
        """
        steering = mv_cmd[0]
        dist = mv_cmd[1]
        if steering > Robot.max_steering_angle:
            raise(ValueError, 'exceeds maximum steering angle')
        if dist < 0:
            raise(ValueError, 'the robot cannot move backwards')
        # with Gaussian noise
        s_noise = np.random.normal(0, self.steering_noise)
        d_noise = np.random.normal(0, self.distance_noise)
        steering = steering if steering + s_noise <= Robot.max_steering_angle else Robot.max_steering_angle
        dist = dist if dist + d_noise >= 0 else 0

        # turning angle
        beta = dist / self.length * np.tan(steering)

        res = Robot()
        res.set_noise(self.bearing_noise, self.steering_noise, self.distance_noise)

        if abs(beta) > 0.001:
            # radius of curvature that rear wheels move along
            r = dist / beta         # same as self.length / np.tan(steering)
            cx = self.x - np.sin(self.orientation) * r
            cy = self.y + np.cos(self.orientation) * r
            res.x = cx + np.sin(self.orientation + beta) * r
            res.y = cy - np.cos(self.orientation + beta) * r
            res.orientation = (self.orientation + beta) % (2 * np.pi)

        else:
            # i.e. almost go straight
            res.x = self.x + dist * np.cos(self.orientation)
            res.y = self.y + dist * np.sin(self.orientation)
            res.orientation = (self.orientation + beta) % (2 * np.pi)

        return res

    def sense(self, add_noise=0):
        """measure the bearing angles between the robot and each landmark"""
        z = []      # bearing angles
        if add_noise:
            b_noise = np.random.normal(0, self.bearing_noise)
        else:
            b_noise = 0

        for mark in Robot.landmarks:
            delta_y = mark[0] - self.y
            delta_x = mark[1] - self.x
            gl_angle = (np.arctan2(delta_y, delta_x) + b_noise) % (2 * np.pi)
            z.append(gl_angle - self.orientation)

        return z    # a vector of bearing angles


class ParticleFilter:
    def __init__(self, n=1000):
        self.num = n        # number of particles
        self.robots = []
        self.initialize()

    def initialize(self):
        # create a list of Robot objects, supposed to be called only once
        for i in range(self.num):
            robo = Robot()
            robo.set_noise(0.1, 0.1, 5)
            self.robots.append(robo)
        return

    def gauss_prob(self, mean, std, x):
        """get a probability value from the Gaussian distribution at given x point"""
        error = (x - mean + np.pi) % (2 * np.pi) - np.pi        # error range: [-pi, pi], instead of [0, 2*pi] !!
        return np.exp(- error**2 / (2 * std**2)) / np.sqrt(2 * np.pi * std**2)

    def set_weight(self, robot, measurement):
        """set importance weight for the input robot which is a particle"""
        pred_zs = robot.sense(0)
        weight = 1
        for pred_z, m in zip(pred_zs, measurement):
            prob = self.gauss_prob(pred_z, robot.bearing_noise, m)
            weight *= prob

        return weight

    def resample(self, weights):
        """
        :param weights: numpy array
        :return:
        """
        index = np.random.randint(0, self.num)
        beta = 0
        res = []        # a new list of particles
        # simulate resampling wheel
        for i in range(self.num):
            beta += np.random.uniform(0, 2 * np.max(weights))
            while weights[index] < beta:
                beta -= weights[index]
                index = (index + 1) % self.num      # wheel
            res.append(self.robots[index])
        return res

    def filter(self, movements, measurements):
        """
        Main workflow
        :param movements: a list of moving commands, each command contains steering angle and distance
        :param measurements: a list of vectors, each vector contains 4 bearing angles
        :return: a vector of three elements [x, y, orientation]
        """
        for u, z in zip(movements, measurements):
            # updating importance weights
            weights = []

            for i in range(self.num):
                self.robots[i] = self.robots[i].move(u)
                weights.append(self.set_weight(self.robots[i], z))

            # normalize weights
            weights = np.array(weights) / sum(weights)

            # resampling
            self.robots = self.resample(weights)

        return self.get_states()

    def get_states(self):
        """get mean states from all current particles"""
        x = 0
        y = 0
        ori = 0
        for r in self.robots:
            x += r.x
            y += r.y
            # orientation is really tricky since it is cyclic.(I am not really understand this part)
            # it seems that this is an approximate version for cheaper computation with acceptable accuracy
            ori += ((r.orientation + np.pi - self.robots[0].orientation) % (2 * np.pi)
                    - np.pi + self.robots[0].orientation)
        return np.array([x, y, ori]) / self.num


def generate_ground_truth(movements):
    """
    in order to test my ParticleFilter
    :param movements: given a sequence of movements
    :return: final robot states and a sequence of measurements corresponding to movements
    """
    myRobot = Robot()       # can be viewed as a single particle for the truth
    myRobot.set_noise(0.1, 0.1, 5)
    zs = []         # measurements
    for u in movements:
        myRobot = myRobot.move(u)
        zs.append(myRobot.sense(0))
    final_states = np.array([myRobot.x, myRobot.y, myRobot.orientation])

    return final_states, zs


def check_output(truth, estimation):
    """
    :param truth: np array
    :param estimation: np array
    :return:
    """
    # tolerance
    tol_xy = 15
    tol_ori = 0.25

    delta_xy = abs(truth[0:2] - estimation[0:2])
    # wrap to delta_orientation range: [-pi, pi] with period of 2*pi;   cyclic radius range
    delta_ori = abs((truth[-1] - estimation[-1] + np.pi) % (2 * np.pi) - np.pi)
    if all(delta_xy < tol_xy) and delta_ori < tol_ori:
        return True
    else:
        return False


count = 0       # number of good results(True)
for i in range(20):
    t1 = time.time()

    # given movements
    num_mvs = 10
    movements = [[np.pi / 6, 10] for row in range(num_mvs)]     # just circular movements
    # truth values
    states_truth, measurements = generate_ground_truth(movements)

    # estimated values
    particle_filter = ParticleFilter(500)
    states_estimation = particle_filter.filter(movements, measurements)

    print('\ntruth: ' + str(states_truth))
    print('estimation: ' + str(states_estimation))
    # check result
    result = check_output(states_truth, states_estimation)
    print('Results check: ' + str(result))

    if result:
        count += 1

    t2 = time.time()
    print('This takes ' + str(t2 - t1))

print(str(count) + ' True out of 20 trails\n')











