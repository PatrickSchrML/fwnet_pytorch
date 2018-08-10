""""
Here we implement a class for loading data.
"""

import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from itertools import cycle, islice


class circles(Dataset):
    def __init__(self, train=True):

        X, y = datasets.make_circles(n_samples=2000, factor=.2, noise=0.1)

        y[y == 0] = -1
        for idx_feature in range(X.shape[1]):
            #X[:, idx_feature] = X[:, idx_feature] - min(X[:, idx_feature])
            X[:, idx_feature] = X[:, idx_feature] / max(X[:, idx_feature])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,
                                                                                y,
                                                                                test_size=0.2,
                                                                                shuffle=True)
        self.data = torch.FloatTensor(self.X_train if train else self.X_test)
        self.labels = torch.FloatTensor(self.y_train if train else self.y_test)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class blobs(Dataset):
    def __init__(self, train=True):

        X, y = datasets.make_blobs(n_samples=1000, random_state=0, centers=2, cluster_std=0.6)

        y[y == 0] = -1
        for idx_feature in range(X.shape[1]):
            X[:, idx_feature] = X[:, idx_feature] - np.mean(X[:, idx_feature])
            X[:, idx_feature] = X[:, idx_feature] / max(X[:, idx_feature])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

        self.data = torch.FloatTensor(self.X_train if train else self.X_test)
        self.labels = torch.FloatTensor(self.y_train if train else self.y_test)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def show_data(data, labels):
    labels = labels.astype(int) + 1
    print(labels)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(labels) + 1))))
    plt.scatter(data[:, 0], data[:, 1], s=10, color=colors[labels])

    #plt.xlim(-2.5, 2.5)
    #plt.ylim(-2.5, 2.5)

    plt.show()

if __name__ == '__main__':
    #dataset = circles()
    dataset = blobs()
    show_data(dataset.data.cpu().detach().numpy(),
              dataset.labels.cpu().detach().numpy())
