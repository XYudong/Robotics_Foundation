import numpy as np

"""
Kalman filter:
Continuous states; Uni-modal; Gaussian distribution
"""


class MultiDKalman():
    def __init__(self):
        self.dt = 0.1       # second
        self.u = np.matrix([[0], [0], [0], [0]])  # external motion(e.g. acceleration)
        self.F = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])  # transition function

        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])  # measurement function
        self.R = np.eye(2) * 0.1  # measurement uncertainty
        self.I_mat = np.eye(4)

        # self.measurements = np.matrix([[5, 10], [6, 8], [7, 6], [8, 4], [9, 2], [10, 0]])
        self.measurements = np.matrix([[1, 4], [6, 0], [11, -4], [16, -8]])
        # location(x, y) in this case

    def kalman_filter(self, x, P):
        """
        :param x: states vector
        :param P: uncertainty covariance matrix
        :return: x, P
        """
        for z in self.measurements:
            x, P = self.predict(x, P)
            print('\npredict: ')
            print(x)
            # print(P)
            x, P = self.update(x, z, P)
            print('update:')
            print(x)
            # print(P)

    def update(self, x, z, P):
        err = z.transpose() - self.H * x      # error

        S = self.H * P * self.H.transpose() + self.R
        K = P * self.H.transpose() * np.linalg.inv(S)      # Kalman gain
        # The Kalman Gain will decrease if the readings (measurements) match the predicted system state.

        # update
        x = x + K * err
        P = (self.I_mat - K * self.H) * P

        return x, P

    def predict(self, x, P):
        x = self.F * x + self.u
        P = self.F * P * self.F.transpose()
        return x, P

    def main(self):
        """
        From this example, we can find that without the measurement of velocity,
        we can still have a good estimation on the state vector(x) through Kalman filter.
        """
        initial_xy = [-4, 8]
        x = np.matrix([[initial_xy[0]], [initial_xy[1]], [0], [0]])     # initial state (x, y, x_dot, y_dot)
        P = np.matrix([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 1000, 0],
                       [0, 0, 0, 1000]])   # initial uncertainty covariance matrix
        self.kalman_filter(x, P)


kk = MultiDKalman()
kk.main()

