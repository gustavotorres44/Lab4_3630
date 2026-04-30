import torch
from torch import nn
import numpy as np

import lab4.infrastructure.pytorch_util as ptu


class MLPPolicy(nn.Module):

    def __init__(self, 
                 ac_dim, 
                 ob_dim,
                 n_layers, 
                 size, 
                 discrete=False, 
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__()

        # parameters for policy architecture
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.discrete = discrete
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.build_graph()

    ##################################

    def build_graph(self):
        if self.discrete:
            self.logits_na = ptu.build_mlp(self.ob_dim, self.ac_dim, self.n_layers, self.size)
            self.logits_na.to(ptu.device)
            self.mean_na = None
            self.logstd_param = None
        else:
            self.logits_na = None
            self.mean_na = ptu.build_mlp(self.ob_dim, self.ac_dim, self.n_layers, self.size)
            self.mean_na.to(ptu.device)
            self.logstd_param = nn.Parameter(torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device))

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def forward(self, obs: torch.Tensor):  # Equivalent to calling `policy(obs)`.
      if self.discrete:
        logits = self.logits_na(obs)
        action = torch.distributions.Categorical(logits=logits).sample()
          
      else:
        mean = self.mean_na(obs)
        logstd = torch.exp(self.logstd_param)
        action = torch.distributions.Normal(mean, logstd).sample()
      return action

    def get_action(self, obs: np.ndarray): # takes a single np observation and returns a single np action
        if len(obs.shape) < 2:
          obs = obs[None]  # Add batch dimension if missing
        
        obs = ptu.from_numpy(obs.astype(np.float32))
        
        # TODO: return the action that the policy prescribes for `obs`.
        # HINT: for discrete policies, sample from the categorical distribution.
        # HINT: for continuous BC evaluation, use the learned mean action directly
        # instead of adding Gaussian exploration noise.
        raise NotImplementedError
        return ptu.to_numpy(action)  # Return with batch dimension, utils.py will extract [0]


#####################################################
#####################################################
class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()
        # Fix 2: initialize the optimizer here, after build_graph() has run
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def update(self, observations, actions, adv_n=None, acs_labels_na=None, qvals=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)

        # TODO: update the policy on the provided supervised batch and return the loss.
        # HINT: supervised BC compares predicted actions to expert labels directly,
        # so train against differentiable network outputs rather than sampled actions.
        raise NotImplementedError
