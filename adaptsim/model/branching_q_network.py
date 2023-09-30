import torch
from torch import nn
import numpy as np
import logging

from adaptsim.model.mlp import MLP


class BranchingQNetwork(nn.Module):

    def __init__(
        self,
        mlp_dim,
        num_bin,  # len = action_dim
        skip_dim=0,
        activation_type='relu',
        use_ln=False,
        device='cpu',
        verbose=True,
    ):

        super().__init__()
        self.device = device
        self.num_bin = num_bin
        self.skip_dim = skip_dim

        self.model = MLP(
            mlp_dim, activation_type, out_activation_type=activation_type,
            use_ln=use_ln, verbose=False
        ).to(device)
        self.value_head = nn.Linear(self.skip_dim + mlp_dim[-1], 1).to(device)

        self.adv_heads = nn.ModuleList([
            nn.Linear(self.skip_dim + mlp_dim[-1], bin_size).to(device)
            for bin_size in num_bin
        ])
        if verbose:
            logging.info("BranchingQ has the architecture as below:")
            logging.info(self.model.moduleList)

    def scale_gradient(self):
        """Scale the gradient of the shared network"""
        for p in self.model.parameters():
            p.grad *= 1 / (len(self.num_bin) + 1)

    def forward(self, x):
        # Convert to torch
        np_input = False
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
            np_input = True

        if self.skip_dim > 0:
            x_skip = x[:, :self.skip_dim]
        out = self.model(x)
        if self.skip_dim > 0:
            out = torch.cat((x_skip, out), dim=1)

        value = self.value_head(out)
        advs = torch.stack([l(out) for l in self.adv_heads], dim=1)
        q_val = value.unsqueeze(2) + advs - advs.mean(2, keepdim=True)

        # Convert back to np
        if np_input:
            q_val = q_val.detach().cpu().numpy()
        return q_val

    def value(self, x):
        # Convert to torch
        np_input = False
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
            np_input = True

        if self.skip_dim > 0:
            x_skip = x[:, :self.skip_dim]
        out = self.model(x)
        if self.skip_dim > 0:
            out = torch.cat((x_skip, out), dim=1)

        value = self.value_head(out)

        # Convert back to np
        if np_input:
            value = value.detach().cpu().numpy()
        return value
