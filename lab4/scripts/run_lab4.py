import os
import time

from lab4.infrastructure.rl_trainer import RL_Trainer
from lab4.agents.bc_agent import BCAgent

class BC_Trainer(object):

    def __init__(self, params):

        #######################
        ## AGENT PARAMS
        #######################

        agent_params = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'max_replay_buffer_size': params['max_replay_buffer_size'],
            }

        self.params = params
        # This lab uses the BC agent, which wraps a supervised MLP policy and replay buffer.
        self.params['agent_class'] = BCAgent
        self.params['agent_params'] = agent_params

        ################
        ## RL TRAINER
        ################

        # Reuse the shared training/logging scaffold for behavior cloning experiments.
        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            n_iter=self.params['n_iter'],
            initial_expertdata=self.params['expert_data'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Train a behavior cloning policy on expert demonstration data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            'Important logged metrics:\n'
            '  Eval_AverageReturn = average episode reward of the current learned policy.\n'
            '  Train_AverageReturn = average return of the trajectories added this iteration.\n'
            '  Initial_DataCollection_AverageReturn = average return of the initial expert dataset.\n'
            'For plain BC, n_iter=1 is usually the right starting point.'
        ),
    )
    parser.add_argument(
        '--expert_data', '-ed',
        type=str,
        required=True,
        help='Path to the expert demonstration .pkl file, relative to where you run the script.',
    )
    parser.add_argument(
        '--env_name', '-env',
        type=str,
        required=True,
        help='Gymnasium environment name, for example Ant-v4 or Hopper-v4.',
    )
    parser.add_argument(
        '--exp_name', '-exp',
        type=str,
        default='pick an experiment name',
        required=True,
        help='Short experiment name used when creating the output log directory.',
    )
    parser.add_argument(
        '--ep_len',
        type=int,
        help='Maximum rollout length. If omitted, uses the environment default episode length.',
    )

    parser.add_argument(
        '--num_agent_train_steps_per_iter',
        type=int,
        default=1000,
        help='Number of gradient updates to run per training iteration.',
    )
    parser.add_argument(
        '--n_iter', '-n',
        type=int,
        default=1,
        help='Number of training iterations. For plain BC, start with 1.',
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Minimum number of environment timesteps to collect when gathering fresh rollouts.',
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=1000,
        help='Minimum number of evaluation timesteps to collect when logging policy performance.',
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=100,
        help='Minibatch size sampled from the replay buffer for each gradient step.',
    )

    parser.add_argument(
        '--n_layers',
        type=int,
        default=2,
        help='Number of hidden layers in the policy network.',
    )
    parser.add_argument(
        '--size',
        type=int,
        default=64,
        help='Width of each hidden layer in the policy network.',
    )
    parser.add_argument(
        '--learning_rate', '-lr',
        type=float,
        default=5e-3,
        help='Learning rate for supervised behavior cloning updates.',
    )

    parser.add_argument(
        '--video_log_freq',
        type=int,
        default=5,
        help='Log rollout videos every N iterations. Use -1 to disable video logging.',
    )
    parser.add_argument(
        '--scalar_log_freq',
        type=int,
        default=1,
        help='Log scalar metrics every N iterations.',
    )
    parser.add_argument(
        '--no_gpu', '-ngpu',
        action='store_true',
        help='Force training onto CPU even if a GPU is available.',
    )
    parser.add_argument(
        '--which_gpu',
        type=int,
        default=0,
        help='GPU id to use when GPU training is enabled.',
    )
    parser.add_argument(
        '--max_replay_buffer_size',
        type=int,
        default=1000000,
        help='Maximum number of transitions stored in the replay buffer.',
    )
    parser.set_defaults(save_params=True)
    parser.add_argument(
        '--save_params',
        dest='save_params',
        action='store_true',
        help='Save policy checkpoints during training (default: enabled)',
    )
    parser.add_argument(
        '--no_save_params',
        dest='save_params',
        action='store_false',
        help='Disable checkpoint saving during training',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed for NumPy, PyTorch, and environment behavior.',
    )
    args = parser.parse_args()

    # convert args to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    logdir_prefix = 'q1_'

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)


    ###################
    ### RUN TRAINING
    ###################

    trainer = BC_Trainer(params)
    trainer.run_training_loop()

if __name__ == "__main__":
    main()
