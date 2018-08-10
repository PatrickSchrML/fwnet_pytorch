import torch.nn as nn
import torch


class FWStackedLayer(nn.Module):
    def __init__(self):
        super(FWStackedLayer, self).__init__()
        # self.

    def forward(self, x):
        # compute gradient

        return output


class FWLayer_svm(nn.Module):
    def __init__(self, num_units, c, gamma, activation_param):
        super(FWLayer_svm, self).__init__()
        self.num_units = num_units
        self.c = c
        self.gamma = gamma
        self.activation_param = activation_param

    def forward(self, learning_rate, state, kernel):
        # compute gradient

        v = torch.matmul(kernel, state)

        ug = torch.mul(v, self.activation_param)

        softmaxed = nn.functional.softmax(torch.transpose(ug, 0, 1), 0)
        g = torch.transpose(softmaxed, 0, 1)

        next_state = torch.add(torch.sub(state, torch.mul(state, learning_rate)),
                               torch.mul(g, learning_rate))

        return next_state, ug, g

        # Linear Kernel:
        # K(x1, x2) = t(x1) * x2

    def compute_linear_kernel(self, X, labels):
        eye = torch.eye(self.num_units)

        c = self.c
        y = labels.view([self.num_units, 1])
        yyT = torch.matmul(y, torch.transpose(y, 0, 1))
        Z = X * y
        kernel = torch.matmul(Z, torch.transpose(Z, 0, 1))
        # Modified Frank-Wolfe Alogrithm for Enhanced Sparsity in Support Vector Machine Classifiers
        return kernel  # + yyT + eye / (2 * c)

        # Gaussian Kernel (RBF):
        # K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)

    def compute_rbf_kernel(self, X, labels):
        # https://github.com/nfmcclure/tensorflow_cookbook/blob/master/04_Support_Vector_Machines/04_Working_with_Kernels/04_svm_kernels.py
        eye = torch.eye(self.num_units)

        gamma = self.gamma  # gamma=-2 activation_param_init=-750 c_init=.75 for circles
        c = self.c

        y = labels.view([self.num_units, 1])
        yyT = torch.matmul(y, torch.transpose(y, 0, 1))

        dist = torch.sum(X.pow(2), 1)
        dist = dist.unsqueeze_(0)
        sq_dists = torch.add(torch.sub(dist, torch.mul(torch.matmul(X, torch.transpose(X, 0, 1)), 2.)),
                             torch.transpose(dist, 0, 1))
        kernel = torch.exp(torch.mul(torch.abs(sq_dists), gamma)) * yyT

        # Modified Frank-Wolfe Alogrithm for Enhanced Sparsity in Support Vector Machine Classifiers
        # return kernel + eye_with_dropout / c
        return kernel + yyT + eye / (2 * c)


class FWLayer_tracenorm(nn.Module):
    def __init__(self, num_units, activation_param, num_batches=1):
        super(FWLayer_tracenorm, self).__init__()

        self._num_units = num_units
        self._activation_param = activation_param
        self._num_batches = num_batches

    def forward(self, inputs, state, X, y):
        # compute gradient
        learning_rate = inputs
        softmaxed = nn.softmax(torch.matmul(X, torch.transpose(state)))
        diff = torch.subtract(softmaxed, y)
        v = torch.matmul(torch.transpose(diff), X)
        ug = torch.multiply(self._activation_param, v)
        g = ug  # tf.transpose(tf.nn.softmax(tf.transpose(ug)))
        # g = tf.nn.softmax(ug)
        # g = ug

        next_state = torch.add(torch.subtract(state, torch.multiply(learning_rate, state)),
                               torch.multiply(learning_rate, g))
        print(next_state)

        return inputs, next_state, ug, g
