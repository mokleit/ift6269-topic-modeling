import numpy as np
from scipy import special
import math


class Utils:
    @classmethod
    def log_gamma(self, x):
        return math.lgamma(x)

    @classmethod
    def tri_gamma(self, x):
        return special.polygamma(2, x)

    @classmethod
    def di_gamma(self, x):
        return special.polygamma(1, x)

    @classmethod
    def log_sum(self, phisum, phi):
        if phisum < phi:
            v = phi + np.log(1 + np.exp(phisum - phi))
        else:
            v = phisum + np.log(1 + np.exp(phi - phisum))


