import numpy as np
from matplotlib import pyplot as plt
import k_means as km
import GMM as gmm


def preprocess_data(file_name):
    """
    Pre-process data, split by space
    :param file_name: string
    :return: ndarray
    """
    output = []
    with open(file_name, 'r') as read_file:
        for input_line in read_file:
            input_line = input_line.strip('\n').split("  ")
            vector = [float(x) for x in input_line]
            output.append(vector)
    return np.array(output, float)


if __name__ == "__main__":
    data = preprocess_data("GMM_dataset.txt")
    # plt.scatter(data[:,0], data[:,1], c='black', s=7)
    # plt.savefig('raw_data.png')
    # k = 10
    # k_cluster = km.KMeans(k, data)
    # k_cluster.best_run(data)
    # k_cluster.iteration_run(data, iterations=10)
    g_model = gmm.GMM(n_clusters=5, data=data)
    g_model.run(data)
