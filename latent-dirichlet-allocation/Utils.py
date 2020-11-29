import math
from scipy import special


class Utils:
    def log_gamma(self, x):
        return math.lgamma(x)

    def tri_gamma(self, x):
        return special.polygamma(2, x)

    def di_gamma(self, x):
        return special.polygamma(1, x)



