"""
From BayesSim

"""
import numpy as np
import torch

from adaptsim.util.dist import Uniform, Gaussian
from adaptsim.util.summarizers import *
from adaptsim.model.mdnn import MDNN  # used dynamically


class BayesSim():

    def __init__(self, cfg):
        """
        Managing inference network for the simulation parameters.
        """
        self.cfg = cfg
        self.device = cfg.device
        self.fine_tune = cfg.fine_tune
        self.output_dim = cfg.model.output_dim
        self.num_train_traj_per_batch = cfg.num_train_traj_per_batch  # was default to 1k
        self.num_train_epochs = 10
        self.batch_size = 100
        self.num_grad_updates = self.num_train_epochs * self.num_train_traj_per_batch // self.batch_size
        self.test_fraction = 0.2

        # Initialize RNG
        self.rng = np.random.default_rng(seed=cfg.seed)

        # None is used in original implementation
        self.prior = None  # None or torch.distributions.MixtureSameFamily
        self.proposal = None

        #? Looks like this is just to get the input dim
        self.summarizer_fxn = eval(cfg.summarizer_fxn.name)
        self.cfg_summarizer_fxn = cfg.summarizer_fxn
        self.cfg.model.device = cfg.device
        self.model_class = cfg.model.name

        # Initialize model
        self.model = eval(self.model_class)(**self.cfg.model)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def reset_model(self, path=None):
        if path is not None:
            self.model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
        elif not self.fine_tune:
            self.model = eval(self.model_class)(**self.cfg.model)

    # @staticmethod
    def get_n_trajs_per_batch(self, n_train_trajs, n_train_trajs_done):
        n_trajs_per_batch = self.num_train_traj_per_batch
        if n_train_trajs_done + n_trajs_per_batch > n_train_trajs:
            n_trajs_per_batch = n_train_trajs - n_train_trajs_done
        return n_trajs_per_batch

    def run_training(self, params, traj_states, traj_actions):
        """Runs the BayesSim algorithm training.

        Parameters
        ----------
        params : torch.Tensor
            Simulation parameters data
        traj_states: torch.Tensor
            Trajectory states
        traj_actions: torch.Tensor
            Trajectory actions

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged during training
        """
        params = params[:, :self.output_dim
                       ]  # assume adapt param is at the front, yikes...
        traj_summaries = self.summarizer_fxn(
            traj_states, traj_actions, self.cfg_summarizer_fxn
        )
        log_dict = self.model.run_training(
            x_data=traj_summaries,
            y_data=params,
            n_updates=self.num_grad_updates,
            batch_size=self.batch_size,
            test_frac=self.test_fraction,
        )
        return log_dict

    def predict(self, states, actions, threshold=0.005):
        """Predicts posterior given x.

        Parameters
        ----------
        states: torch.Tensor
            Trajectory states
        actions: torch.Tensor
            Trajectory actions
        threshold: float (optional)
            A threshold for pruning negligible mixture components.

        Returns
        -------
        posterior : MoG
            A mixture posterior
        """
        xs = self.summarizer_fxn(states, actions, self.cfg_summarizer_fxn)

        if self.model_class == 'PointNN':
            mu = self.model.forward(xs).flatten().detach().cpu().numpy(
            )  # normalized in [0,1]
            output_lows = self.model.output_lows.detach().cpu().numpy()
            output_highs = self.model.output_highs.detach().cpu().numpy()
            mu = output_lows + mu * (
                output_highs-output_lows
            )  # unnormalize from [0,1] to [low,high]
            return mu
        else:
            mogs = self.model.predict_MoGs(xs)
            if self.proposal is not None:
                # Compute posterior given prior by analytical division step.
                for tmp_i, mog in enumerate(mogs):
                    mog.prune_negligible_components(threshold=threshold)
                    if isinstance(self.prior, Uniform):
                        post = mog / self.proposal
                    elif isinstance(self.prior, Gaussian):
                        post = (mog * self.prior) / self.proposal
                    else:
                        raise NotImplemented
                    mogs[tmp_i] = post

            if len(mogs) == 1:
                return mogs[0]

            # Make a small net to fit the combined mixture.
            kwargs = {
                'input_dim': 1,  # unconditional MDNN
                'output_dim': self.model.output_dim,
                'output_lows': self.model.output_lows.detach().cpu().numpy(),
                'output_highs': self.model.output_highs.detach().cpu().numpy(),
                'n_gaussians': self.model.n_gaussians,
                'hidden_layers': (128, 128),
                'lr': self.model.lr,
                'activation': self.model.activation,
                'full_covariance': self.model.L_size > 0,
                'device': self.model.device
            }
            mog_model = MDNN(**kwargs)

            # Re-sample MoGs.
            mog_smpls = None
            tot_smpls = int(1e4)
            n_smpls_per_mog = int(tot_smpls / xs.shape[0])
            for tmp_i in range(xs.shape[0]):
                new_smpls = mogs[tmp_i].gen(
                    self.rng, n_samples=n_smpls_per_mog
                )
                if mog_smpls is None:
                    mog_smpls = new_smpls
                else:
                    mog_smpls = np.concatenate([mog_smpls, new_smpls], axis=0)
            mog_smpls = torch.from_numpy(mog_smpls).float().to(
                self.model.device
            )

            # Fit a single MoG to compute the final posterior.
            print(f'Fitting posterior from {len(mogs):d} mogs')
            batch_size = 100
            n_updates = 5 * tot_smpls // batch_size
            input = torch.zeros(mog_smpls.shape[0], 1).to(self.model.device)
            mog_model.run_training(input, mog_smpls, n_updates, batch_size)
            fitted_mogs = mog_model.predict_MoGs(input[0:1, :])
            assert (len(fitted_mogs) == 1)

            return fitted_mogs[0]
