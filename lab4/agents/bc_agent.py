from lab4.infrastructure.replay_buffer import ReplayBuffer
from lab4.policies.MLP_policy import MLPPolicySL
from .base_agent import BaseAgent


class BCAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(BCAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = MLPPolicySL(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self, observations, expert_actions, rewards, next_ob_no, terminal_n):
        # TODO: train the BC actor on the sampled batch and return the training log.
        # HINT: BC only needs (observation, expert action) pairs, even though
        # rewards/next observations/terminals are passed through for interface compatibility.
        raise NotImplementedError

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        # TODO: sample a uniformly random supervised minibatch from the replay buffer.
        raise NotImplementedError

    def save(self, path):
        return self.actor.save(path)
