import numpy as np


class OneDKalman:
    """1D Kalman filter"""
    def update(self, mean0, var0, mean1, var1):
        """measurement"""
        new_mean = (var0 * mean1 + var1 * mean0) / (var0 + var1)
        new_var = var0 * var1 / (var0 + var1)

        return new_mean, new_var

    def predict(self, mean0, var0, mean1, var1):
        """motion"""
        new_mean = mean0 + mean1
        new_var = var0 + var1
        return new_mean, new_var

    def main(self):
        measurements = [5, 6, 7, 9, 10]    # mean of measurement
        motion = [1, 1, 2, 1, 1]   # mean of motion command
        measurement_sig2 = 4
        motion_sig2 = 2
        # initial distribution
        mu = 0
        sig2 = 10000

        for z, u in zip(measurements, motion):
            mu, sig2 = self.update(mu, sig2, z, measurement_sig2)
            print('\nupdate: ' + str([mu, sig2]))

            mu, sig2 = self.predict(mu, sig2, u, motion_sig2)
            print('predict: ' + str([mu, sig2]))


class MultiDKalman():
    def __init__(self):
        self.u = np.matrix([[0], [0]])  # external motion
        self.F = np.matrix([[1, 1], [0, 1]])  # transition function

        self.H = np.matrix([[1, 0]])  # measurement function
        self.R = np.matrix([[1]])  # measurement uncertainty
        self.I_mat = np.identity(2)

        self.measurements = [1, 2, 3]   # of only location in this case

    def kalman_filter(self, x, P):
        """
        :param x: states vector
        :param P: uncertainty covariance
        :return: x, P
        """
        for z in self.measurements:
            x, P = self.update(x, z, P)
            print('\nupdate:')
            print(x)
            print(P)
            x, P = self.predict(x, P)
            print('predict: ')
            print(x)
            print(P)

    def update(self, x, z, P):
        err = z - self.H * x      # error
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
        From this example, we can find that without the measurement of velocity(i.e. x[1]),
         we can still have a good estimation on the state vector(x) through Kalman filter.
        """
        x = np.matrix([[0], [0]])     # initial state (location and velocity)
        P = np.matrix([[1000, 0], [0, 1000]])   # initial uncertainty
        self.kalman_filter(x, P)


kk = MultiDKalman()
kk.main()