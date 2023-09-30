import numpy as np
import logging
import torch
from torch.optim import AdamW


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class DynFit():
    """
    Map action to pos
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed=cfg.seed)
        self.input_dim = cfg.arch.mlp[0]
        self.hidden_dim = cfg.arch.mlp[1]
        self.output_dim = cfg.arch.mlp[2]
        self.lr = cfg.lr
        self.goal_range = np.array(cfg.goal_range)
        self.action_range = np.array(cfg.action_range)

    def __call__(self, x):
        """X is normalized goal since it is from venv. Return normalized action.
           Policy output is unnormalized goal.
        """
        if len(x.shape) == 1:
            x = x[np.newaxis, ...]
        N = x.shape[0]

        # Unnormalize x from [0,1] to goal_range
        x = x * (self.goal_range[1] - self.goal_range[0]) + self.goal_range[0]

        # Create evenly spaced 2d grid between -1 and 1 and make it 2D array with 2 columns
        actions = np.mgrid[
            -1:1:50j,
            -1:1:50j].reshape(2, -1).T  # action is usually normalized as tanh

        # repeat actions for batch size N in a new dimension at front
        actions = np.repeat(actions[np.newaxis, ...], N, axis=0)

        # forward pass
        goals = self.network(torch.from_numpy(actions).float())  # unnomalized

        # tile x in a new, second dimension
        x = np.tile(x[:, np.newaxis, :], (1, goals.shape[1], 1))

        # find the closest action to the passed in x
        dist = np.linalg.norm(goals - x, axis=-1)
        idx = np.argmin(dist, axis=-1)

        # Index actions with indices at second dimension
        actions_nearest = actions[np.arange(N), idx]
        return actions_nearest  # normalized

    def build_network(self, build_optimizer=True, verbose=True):
        self.network = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.output_dim),
            # torch.nn.Sigmoid()
        )  # output unnormalized positions

        # Create optimizer
        if build_optimizer:
            logging.info("Build optimizer for inference.")
            self.build_optimizer()

    def build_optimizer(self):
        self.optimizer = AdamW(
            self.network.parameters(), lr=self.lr, weight_decay=1e-4
        )

    def update(self, states, actions, batch_size, num_update):
        """Input: both states and actions are unnormalized since they are from collect_data(). Policy output is unnormalized goal."""
        # Normalize actions to [-1,1]
        actions = ((actions - self.action_range[0]) /
                   (self.action_range[1] - self.action_range[0]) * 2
                   - 1).float()

        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            torch.Tensor(actions), torch.Tensor(states)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Train network
        for _ in range(num_update):
            for batch in dataloader:
                action, state = batch
                self.optimizer.zero_grad()
                pred = self.network(action.float())
                loss = torch.nn.functional.mse_loss(pred, state.float())
                loss.backward()
                self.optimizer.step()
                print('loss: ', loss.item())
