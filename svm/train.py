import sys

sys.path.append("/home/patrick/repositories/fwnet_pytorch")
from networks.fwnet import FWLayer_svm
from datasets.datasets import circles, blobs
from torch.utils.data import DataLoader
import torch
import numpy as np
from helper import svm_helper
import matplotlib.pyplot as plt


class Trainer_FW:
    def __init__(self, n_epochs=1):
        self.n_epochs = n_epochs

        #self.dataset = blobs()
        self.dataset = circles()

        self.batch_size = len(self.dataset)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)

        self.kernel_type = "rbf"
        self.c, self.gamma, self.activation_param = 1., -50., -10.
        _init_state = np.random.uniform(low=0, high=1, size=(len(self.dataset), 1))
        _init_state = np.multiply(1 / np.sum(_init_state), _init_state)
        self.state = torch.FloatTensor(_init_state)
        self.network = FWLayer_svm(len(self.dataset), self.c, self.gamma, self.activation_param)

    def train(self):

        if self.kernel_type == "linear":
            kernel = self.network.compute_linear_kernel(self.dataset.data, self.dataset.labels)
        else:
            kernel = self.network.compute_rbf_kernel(self.dataset.data, self.dataset.labels)

        for epoch in range(self.n_epochs):
            for i_batch, (batch_x, batch_y) in enumerate(self.dataloader):
                next_state, ug, g = self.network(learning_rate=0.1, state=self.state, kernel=kernel)
                self.state.data.copy_(next_state)

        # result
        final_state = self.state.data.cpu().detach().numpy()

        #print(final_state)
        xx, yy = np.meshgrid(np.arange(-1.5, 1.5, 0.02),
                             np.arange(-1.5, 1.5, 0.02))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        plt_x = self.dataset.data.data.cpu().detach().numpy()
        plt_y = self.dataset.labels.data.cpu().detach().numpy()
        if self.kernel_type == "linear":
            print(svm_helper.get_acc(final_state,
                                     plt_x, plt_y,
                                     plt_x, plt_y,
                                     self.c, self.gamma, len(self.dataset), 0,
                                     kernel_type=self.kernel_type))
            grid_predictions = svm_helper.linear_pred(final_state, plt_x, plt_y,
                                                      grid_points,
                                                      len(self.dataset))
        else:
            grid_predictions = svm_helper.rbf_pred(final_state, plt_x, plt_y,
                                                   grid_points,
                                                   self.gamma, len(self.dataset))
        grid_predictions = grid_predictions.reshape(xx.shape)
        # Plot points and grid
        X_class0_plt = list(zip(*plt_x[plt_y == -1]))
        X_class1_plt = list(zip(*plt_x[plt_y == 1]))

        plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
        plt.plot(X_class0_plt[0], X_class0_plt[1], 'ro', label='Class 1')
        plt.plot(X_class1_plt[0], X_class1_plt[1], 'kx', label='Class -1')
        plt.title('Gaussian SVM Results')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='lower right')
        plt.ylim([-1.5, 1.5])
        plt.xlim([-1.5, 1.5])
        plt.show()


if __name__ == '__main__':
    trainer = Trainer_FW(n_epochs=100)
    trainer.train()
