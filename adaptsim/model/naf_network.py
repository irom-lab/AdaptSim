"""
Normalized Advantage Functions (NAF) network. Shane et al, ?

"""
import torch
from torch import nn
from torch.distributions import MultivariateNormal


class NAF_Network(nn.Module):

    def __init__(
        self,
        seed,
        mlp_dim,
        noise_scale=1,
        device='cpu',
        verbose=False,
    ):
        super(NAF_Network, self).__init__()
        assert len(mlp_dim) == 3  # hidden size used for two layers
        self.seed = torch.manual_seed(seed)
        self.input_shape = mlp_dim[0]
        layer_size = mlp_dim[1]
        self.action_size = mlp_dim[-1]
        self.noise_scale = noise_scale
        self.verbose = verbose

        self.head_1 = nn.Linear(self.input_shape, layer_size).to(device)
        self.ff_1 = nn.Linear(layer_size, layer_size).to(device)
        self.action_values = nn.Linear(layer_size, self.action_size).to(device)
        self.value = nn.Linear(layer_size, 1).to(device)
        self.matrix_entries = nn.Linear(
            layer_size, int(self.action_size * (self.action_size + 1) / 2)
        ).to(device)

    def get_action(self, state, append, noise=False):
        x = torch.relu(self.head_1(state))
        x = torch.relu(self.ff_1(x))
        action_value = torch.tanh(self.action_values(x))

        if noise:
            entries = torch.tanh(self.matrix_entries(x))
            L = torch.zeros(
                (state.shape[0], self.action_size, self.action_size)
            ).to(state.device)
            tril_indices = torch.tril_indices(
                row=self.action_size, col=self.action_size, offset=0
            )
            L[:, tril_indices[0], tril_indices[1]] = entries
            L.diagonal(dim1=1, dim2=2).exp_()
            P = L * L.transpose(2, 1)
            noise_cov = torch.inverse(P) * self.noise_scale
            if self.verbose:
                print('Noise covariance: ', noise_cov)

            dist = MultivariateNormal(action_value, noise_cov)
            action = dist.sample()
            action_value = torch.clamp(action, min=-1, max=1)
        return action_value

    def get_value(self, state, action):

        x = torch.relu(self.head_1(state))
        x = torch.relu(self.ff_1(x))
        action_value = torch.tanh(self.action_values(x)).unsqueeze(-1)
        entries = torch.tanh(self.matrix_entries(x))
        V = self.value(x)

        # create lower-triangular matrix
        L = torch.zeros((state.shape[0], self.action_size, self.action_size)
                       ).to(state.device)

        # get lower triagular indices
        tril_indices = torch.tril_indices(
            row=self.action_size, col=self.action_size, offset=0
        )

        # fill matrix with entries
        L[:, tril_indices[0], tril_indices[1]] = entries
        L.diagonal(dim1=1, dim2=2).exp_()

        # calculate state-dependent, positive-definite square matrix
        P = L * L.transpose(2, 1)
        A = (
            -0.5 * torch.matmul(
                torch.matmul((action.unsqueeze(-1)
                              - action_value).transpose(2, 1), P),
                (action.unsqueeze(-1) - action_value)
            )
        ).squeeze(-1)
        Q = A + V
        return Q, V
