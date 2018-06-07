import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl


class GMM:
    def __init__(self, n_clusters, data):
        self.K = n_clusters
        # initial means
        self.mu = []
        random_index = np.random.randint(0, len(data)-1, size=self.K)
        for idx in random_index:
            self.mu.append(data[idx])
        self.mu = np.array(self.mu, float)

        # initial covariance matrices
        self.sigma = []
        for i in range(self.K):
            self.sigma.append(np.cov(data.T))
        self.sigma = np.array(self.sigma, float)

        # initial mixing coefficients
        self.pi = np.ones(self.K)/self.K

    def gaussian(self, x, k):
        """
        Calculate the Gaussian distribution for x_K, mu_K, and sigma_K
        :param x: ndarray
        :param k: int - which is the k of the cluster currently being considered
        :return: float - the Gaussian probability
        """
        D = len(x)
        normalization_factor = 1.0/np.sqrt((2.0*np.pi)**D * np.linalg.det(self.sigma[k]))
        x_mu = np.matrix(x - self.mu[k], float)
        dot_product = x_mu * np.linalg.inv(self.sigma[k]) * x_mu.T
        return normalization_factor * np.exp(-0.5 * dot_product)

    def likelihood(self, x):
        """
        Calculate the likelihoods of x
        :param x: ndarray
        :return: float - the likelihood
        """
        result = 0.0
        for k in range(self.K):
            result += self.pi[k] * self.gaussian(x, k)
        return result

    def e_step(self, data):
        """
        Calculate posterior responsibilities of all data points at each K
        :param data: ndarray
        :return: responsibilities: ndarray, likelihoods: list
        """
        responsibilities = np.zeros((len(data), self.K))
        for idx in range(len(data)):
            llh_i = self.likelihood(data[idx])
            for k in range(self.K):
                responsibilities[idx][k] = (self.pi[k] * self.gaussian(data[idx], k)) / llh_i
        return responsibilities

    def m_step(self, responsibilities, data):
        """
        Update parameters: means, covariance matrices, mixing coefficients
        :param responsibilities: ndarray
        :param data: ndarray
        """
        self.mu = np.zeros(self.mu.shape)
        self.sigma = np.zeros(self.sigma.shape)
        for k in range(self.K):
            N_k = np.sum(responsibilities[:, k])
            # update mu - mean
            for idx in range(len(data)):
                self.mu[k] += responsibilities[idx][k] * data[idx]
            self.mu[k] /= N_k
            # update sigma - covariance matrix
            for idx in range(len(data)):
                x_mu = np.array([data[idx]]) - np.array([self.mu[k]])
                self.sigma[k] += responsibilities[idx][k] * x_mu * x_mu.T
            self.sigma[k] /= N_k
            # update pi - mixing coefficient
            self.pi[k] = N_k/len(data)

    def run(self, data):
        """
        Run the EM for 20 iterations, save the plot for each iteration
        :param data: ndarray
        """
        labels = np.zeros(len(data))
        colors = np.random.rand(self.K, 3)
        for count in range(50):
            responsibilities = self.e_step(data)
            self.m_step(responsibilities, data)
            for idx in range(len(data)):
                labels[idx] = np.argmax(responsibilities[idx])
            self.plot_results(data, labels, "gmm_{}".format(count), colors)

    def plot_results(self, data, label, title, color_iter):
        """
        Plot the result
        :param data: ndarray
        :param label: list
        :param title: string
        :param color_iter: ndarray
        """
        fig, ax = plt.subplots()
        for i, (mean, covar, color) in enumerate(zip(self.mu, self.sigma, color_iter)):
            v, w = np.linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(label == i):
                continue
            ax.scatter(data[label == i, 0], data[label == i, 1], s=7, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

        ax.set_title(title)
        fig.savefig("gmm_plots/{}.png".format(title))
