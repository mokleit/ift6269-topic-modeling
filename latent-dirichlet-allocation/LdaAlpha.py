import numpy as np
from Utils import Utils


class LdaAlpha:
    def __init__(self):
        self.NEWTON_THRESH = 1e-5
        self.MAX_ALPHA_ITER = 1000

    # Objective function L: double a, double ss, int D, int K
    def alhood(self, a, ss, D, K):
        factor = Utils.log_gamma(K * a) - Utils.log_gamma(a)
        return D * factor + (a - 1) * ss

    # First derivative of L: double a, double ss, int D, int K
    def d_alhood(self, a, ss, D, K):
        factor = (K * Utils.di_gamma(K * a) - K * Utils.di_gamma(a))
        return D * factor + ss

    # Second derivative of L: double a, int D, int K
    def d2_alhood(self, a, D, K):
        factor = (K * K * Utils.tri_gamma(K * a) - K * Utils.tri_gamma(a))
        return D * factor

    # Implement Newton's method
    def opt_alpha(self, ss, D, K):
        a, log_a, init_a = 100
        iter = 0
        while True:
            iter += 1
            a = np.exp(log_a)
            if np.isnan(a):
                init_a = init_a * 10
                print("WARNING: alpha is NaN. New init = ", init_a)
                a = init_a
                log_a = np.log(a)
            f = self.alhood(a, ss, D, K)
            df = self.d_alhood(a, ss, D, K)
            d2f = self.d2_alhood(a, D, K)
            log_a = log_a - (df / (d2f * a + df))
            print("Maximisation of alpha: %s  %s" % (f, df))

            if np.abs(df) > self.NEWTON_THRESH and iter < self.MAX_ALPHA_ITER:
                break

        return np.exp(log_a)
