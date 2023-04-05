"""Utility functions from computing recursive (online) covariance matrices
"""

import numpy as np


def mean_tp1(mean_t: np.ndarray, x: np.ndarray, t: float) -> np.ndarray:
    """Return an online mean at t+1

    Args:
        mean_t: Mean at time t
        x: New observation of same shape as mean_t
        t: Value of time t

    Returns:
        The new mean at t+1
    """
    return t / (t + 1.0) * mean_t + 1.0 / (t + 1.0) * x


def cov_tp1(cov_t: np.ndarray, x: np.ndarray, mean_tp1: np.ndarray, t: float) -> np.ndarray:
    """Return an online covariance at t+1

    Args:
        cov_t: Covariance at time t of shape (d,d)
        x: New observation of shape (d)
        mean_tp1: Mean at t+1 (of shape(d))
        t: Time t

    Returns:
        Covariance at t+1 (of shape (d,d))
    """
    x_delta = (x - mean_tp1).reshape(len(x), 1)
    return t / (t + 1.0) * cov_t + (1.0 / t) * (x_delta @ x_delta.T)


class RunningCovariance:
    """Running (online) covariance calculator"""

    def __init__(self, init_cov: np.ndarray, init_mean: np.ndarray, init_t: float):
        """

        Args:
            init_cov: Initial covariance of shape (d,d)
            init_mean: Initial mean of shape (d)
            init_t: Time t for the initial values
        """
        self.cov_t = init_cov
        self.mean_t = init_mean
        self.t = init_t
        self.window = None

    def update(self, x: np.ndarray) -> np.ndarray:
        """Update the internal state with a new observation

        Args:
            x: Array of shape (d) to 'append' to the running calculations

        Returns:
            The updated covariance at time t+1
        """
        if self.window:
            t = self.window
        else:
            t = self.t
        self.mean_t = mean_t1 = mean_tp1(self.mean_t, x, t)
        self.cov_t = cov_tp1(self.cov_t, x, mean_t1, t)
        self.t += 1
        return self.cov_t
