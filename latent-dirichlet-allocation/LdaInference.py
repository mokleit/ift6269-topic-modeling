import numpy as np
from .Utils import Utils
from .LdaDataclasses import Settings


class LdaInference:

    def lda_inference(self, doc, model, var_gamma, phi):
        converged = 1
        phisum = likelihood = 0
        likelihood_old = 0
        num_topics = model.num_topics
        doc_length = doc.length
        old_phi = np.zeros(num_topics)
        di_gamma_gam = np.zeros(num_topics)

        # We now compute posterior dirichlet
        for i in range(num_topics):
            var_gamma[i] = model.alpha + doc.total / num_topics.astype(float)
            di_gamma_gam[i] = Utils.di_gamma(var_gamma[i])
            for j in range(doc_length):
                phi[j][i] = 1.0 / num_topics
        var_iter = 0

        while converged > Settings.VAR_CONVERGED and (var_iter < Settings.VAR_MAX_ITER or Settings.VAR_MAX_ITER == -1):
            var_iter += 1
            for i in range(doc_length):
                phisum = 0
                for j in range(num_topics):
                    old_phi[j] = phi[i][j]
                    phi[i][j] = di_gamma_gam[j] + model.log_prob_w[j][doc.words[i]]

                    if j > 0:
                        phisum = Utils.log_sum(phisum, phi[i][j])
                    else:
                        phisum = phi[i][j]

                for k in range(num_topics):
                    phi[i][k] += np.exp(phi[i][k] - phisum)
                    var_gamma[k] = var_gamma[k] + doc.counts[i] * (phi[i][k] - old_phi[k])
                    di_gamma_gam[k] = Utils.digamma(var_gamma[k])
                    likelihood = self.compute_likelihood(doc, model, phi, var_gamma)
                    converged = (likelihood_old - likelihood) / likelihood_old
                    likelihood_old = likelihood

        return likelihood

    def compute_likelihood(self, doc, model, phi, var_gamma):
        likelihood = digsum = var_gamma_sum = 0
        num_topics = model.num_topics
        dig = np.zeros(num_topics)

        for k in range(num_topics):
            dig[k] = Utils.di_gamma(var_gamma[k])
            var_gamma_sum += var_gamma[k]

        digsum = Utils.di_gamma(var_gamma_sum)
        likelihood = model.alpha * num_topics - model.num_topics * Utils.log_gamma(model.alpha) - Utils.log_gamma(
            var_gamma_sum)

        for k in range(num_topics):
            likelihood += (model.alpha - 1) * (dig[k] - digsum) + Utils.log_gamma(var_gamma[k]) - (var_gamma[k] - 1) * (
                        dig[k] - digsum)
            for l in range(doc.length):
                if phi[l][k] > 0:
                    likelihood += doc.counts[l] * (
                                phi[l][k] * ((dig[k] - digsum) - np.log(phi[l][k]) + model.log_prob_w[k][doc.words[l]]))

        return likelihood
