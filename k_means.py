import numpy as np
import random
import copy
from matplotlib import pyplot as plt


class KMeans:
    def __init__(self, k, data):
        self.k = k
        random_index = np.random.randint(0, len(data)-1, size=self.k)
        self.centroids = []
        for idx in random_index:
            self.centroids.append(data[idx])
        self.centroids = np.array(self.centroids, float)

    def best_run(self, data):
        """
        Run for the best clustering, iterate till the error is 0
        :param data: ndarray
        """
        old_centroids = np.zeros(self.centroids.shape)
        clusters = np.zeros(len(data))
        error = np.linalg.norm(self.centroids - old_centroids)

        while error != 0.0:
            for idx in range(len(data)):
                distance_i = np.linalg.norm(data[idx] - self.centroids, axis=1)
                cluster_i = np.argmin(distance_i)  # take the minimal distance
                clusters[idx] = cluster_i

            old_centroids = copy.deepcopy(self.centroids)

            for label in range(self.k):
                # get points in the recent cluster
                points = [data[j] for j in range(len(data)) if clusters[j] == label]
                # update centroids
                self.centroids[label] = np.mean(points, axis=0)
            error = np.linalg.norm(self.centroids - old_centroids)

        colors = []
        for i in range(self.k):
            colors.append(np.random.shuffle([255,0,0,]))
        fig, ax = plt.subplots()
        for i in range(self.k):
            points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])
            ax.scatter(points[:,0], points[:,1], s=7, color=colors[i])
        ax.scatter(self.centroids[:,0], self.centroids[:,1], marker="*", s=20, color='#050505')
        ax.set_title("Error = {}".format(error))
        fig.savefig("k_means_plots/best_k_mean.png")

    def iteration_run(self, data, iterations=1):
        """
        Iterate the clustering for r times, plot of each run is saved
        :param data: ndarray
        :param iterations: integer
        """
        colors = []
        for i in range(self.k):
            colors.append(np.random.shuffle([255,0,0]))
        for it in range(iterations):
            random_index = np.random.randint(0, len(data)-1, size=self.k)
            self.centroids = []
            for idx in random_index:
                self.centroids.append(data[idx])
            self.centroids = np.array(self.centroids, float)
            old_centroids = np.zeros(self.centroids.shape)
            clusters = np.zeros(len(data))
            error = np.linalg.norm(self.centroids - old_centroids)

            for idx in range(len(data)):
                # E-step
                distance_i = np.linalg.norm(data[idx] - self.centroids, axis=1)
                cluster_i = np.argmin(distance_i)
                clusters[idx] = cluster_i

                old_centroids = copy.deepcopy(self.centroids)

                # M-step
                # get points in the recent cluster
                points = [data[j] for j in range(len(data)) if clusters[j] == cluster_i]
                # update centroids
                self.centroids[cluster_i] = np.mean(points, axis=0)

                error = np.linalg.norm(self.centroids - old_centroids)

            # plot and save the clutering result
            fig, ax = plt.subplots()
            for i in range(self.k):
                points = np.array([data[j] for j in range(len(data)) if clusters[j] == i])
                ax.scatter(points[:,0], points[:,1], s=7, color=colors[i-len(colors)])
            ax.scatter(self.centroids[:,0], self.centroids[:,1], marker="*", s=20, color='#050505')
            print(error)
            ax.set_title("Error = {}".format(error))
            fig.savefig("k_means_plots/km_{}.png".format(it))
