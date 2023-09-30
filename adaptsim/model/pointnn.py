# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Point Estimate NN models for BayesSim."""

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn


class PointNN(nn.Module):
    LL_LIMIT = 1.0e5  # limit log likelihood to avoid large gradients
    MIN_WEIGHT = 1.0e-5  # minimum component weights to enable updates
    EPS_NOISE = 1.e-5  # small noise e.g. for numerical stability

    def __init__(
        self,
        input_dim,
        output_dim,
        output_lows,
        output_highs,
        n_gaussians,
        full_covariance,
        hidden_layers,
        lr,
        activation=torch.nn.Tanh,
        device='cpu',
        **kwargs,
    ):
        """Constructs and initializes a Mixture Density Network.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input
        output_dim : int
            Dimensionality of the output
        output_lows: array
            Flat array of lows for output ranges
        output_highs: array
            Flat array of highs for output ranges
        n_gaussians : int
            Number of Gaussian components for the mixture
        full_covariance : bool
            Whether Gaussian components should be full covariance
        hidden_layers : list or tuple of int
            Size of each fully-connected hidden layer for the main NN
        activation: Module
            torch.nn activation class, e.g. Tanh, LeakyReLU
        lr: float
            Learning rate for the optimizer
        device : string
            Device string (e.g. 'cpu', 'cuda:0')
        """
        super(PointNN, self).__init__()
        self.output_dim = output_dim
        self.output_lows = None
        self.output_highs = None
        if output_lows is not None:
            self.output_lows = torch.tensor(output_lows).float().to(device)
            self.output_highs = torch.tensor(output_highs).float().to(device)
        self.activation = activation
        self.lr = lr
        self.device = device
        # Construct the main part of NN as nn.Sequential
        # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        net = OrderedDict()
        last_layer_size = input_dim
        for l, layer_size in enumerate(hidden_layers):
            net['fcon%d' % l] = nn.Linear(last_layer_size, layer_size)
            net['nl%d' % l] = activation()
            last_layer_size = layer_size
        self.net = nn.Sequential(net) if len(hidden_layers) > 0 else None
        self.mu = nn.Linear(last_layer_size, output_dim * n_gaussians)  # means
        self.sigmoid = torch.nn.Sigmoid()  # normalize mu output
        self.to(self.device)  # move all member variables to device

        self.n_gaussians = 1

    def forward(self, x):
        """Applies NNs to the input and outputs weight, mean, variance info.

        Parameters
        ----------
        x : torch.Tensor
            A batch of input vectors

        Returns
        -------
        weights: torch.Tensor
            Mixture weights
        mu: torch.Tensor
            Mixture means
        L_d: torch.Tensor
            Covariance diagonals for each component
        L: torch.Tensor
            Lower triangular factors for each component
        """
        net_out = self.net(x) if self.net is not None else x
        mu = self.mu(net_out).reshape(-1, self.output_dim)  # not normalized
        mu = self.sigmoid(mu)
        assert (torch.isfinite(mu).all())
        return mu

    def loss_fn(self, mu, y):
        """Computes loss for training MDN.

        Parameters
        ----------
        weights: torch.Tensor
            Mixture weights
        mu: torch.Tensor
            Mixture means
        L_d: torch.Tensor
            Covariance diagonals for each component
        L: torch.Tensor
            Lower triangular factors for each component
        y: torch.Tensor
            target values

        Returns
        -------
        loss: torch.Tensor
            Loss for training MDN.
        """
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(mu, y)
        return loss.mean()

    def run_training(
        self, x_data, y_data, n_updates, batch_size, test_frac=0.2
    ):
        """Runs MDN training.

        Parameters
        ----------
        x_data : torch.Tensor
            Input data
        y_data: torch.Tensor
            Target output data
        n_updates: int
            Number of gradient steps/updates used for neural network training
        batch_size: int
            Batch size for neural network training
        test_frac: float
            Fraction of dataset to keep as test

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged during training
        """
        assert (x_data.shape[0] == y_data.shape[0])
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        y_data = self.normalize_samples(y_data)  # label normalized
        n_tot = x_data.shape[0]
        n_train = max(int(n_tot * (1.0-test_frac)), 1)
        x_train_data = x_data[:n_train].to(self.device)
        y_train_data = y_data[:n_train].to(self.device)
        x_test_data = x_data[n_train:].to(self.device)
        y_test_data = y_data[n_train:].to(self.device)

        # We retain all training data in memory to avoid moving out of GPU.
        # PyTorch implementations often use a data loader that keeps large
        # datasets on disk/CPU and then loads them into GPU memory during
        # training. This could be appropriate for simulators that run on CPU,
        # but for IG we would like to avoid this overhead, since IG simulation
        # can be run directly on the GPU.
        def batch_generator():
            while True:
                ids = np.random.randint(0, len(x_train_data), batch_size)
                yield x_train_data[ids], y_train_data[ids]

        batch_gen_iter = batch_generator()

        train_loss_list = []
        test_loss_list = []
        for epoch in range(n_updates):
            x_batch, y_batch = next(batch_gen_iter)
            optimizer.zero_grad()

            mu = self(x_batch)
            loss = self.loss_fn(mu, y_batch)

            loss.backward()
            optimizer.step()
            if epoch % max(n_updates // 5, 1) == 0 or epoch + 1 == n_updates:
                mu_test = self(x_test_data)
                test_loss = self.loss_fn(mu_test, y_test_data)
                test_loss = test_loss.item()
                train_loss_list.append(loss.item())
                test_loss_list.append(test_loss)
                print(
                    f'loss: train {loss.item():0.4f}'
                    f' test {test_loss:0.4f}'
                )
        return {'train_loss': train_loss_list, 'test_loss': test_loss_list}

    def normalize_samples(self, params):
        rng = self.output_highs - self.output_lows
        normed_params = (params - self.output_lows) / rng
        return normed_params
