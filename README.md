## Setup

1. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

1. install the zip file from canvas
1. upload the zipped folder to your Drive, under directly "My Drive"
1. unzip "lab4_release.zip" 
1. open your web browser and navigate to google drive
1. drag run_lab4_colab.ipynb file directly to the Drive
1. this should open the python notebook in google colab
1. run all cells in setup to make sure the requirements for the enviornment are correclty set up

## Complete the code

Fill in sections marked with `TODO`. In particular, see
 - [infrastructure/rl_trainer.py](lab4/infrastructure/rl_trainer.py)
 - [policies/MLP_policy.py](lab4/policies/MLP_policy.py)
 - [infrastructure/replay_buffer.py](lab4/infrastructure/replay_buffer.py)
 - [infrastructure/utils.py](lab4/infrastructure/utils.py)
 - [infrastructure/pytorch_util.py](lab4/infrastructure/pytorch_util.py)

Look for sections marked with `TODO` to see how the edits you make will be used.
Some other files that you may find relevant
 - [scripts/run_lab4.py](lab4/scripts/run_lab4.py)
 - [agents/bc_agent.py](lab4/agents/bc_agent.py)

See the homework pdf for more details.

## Implementation Guide

Recommended order:
 - Start with [infrastructure/pytorch_util.py](lab4/infrastructure/pytorch_util.py) and build the MLP first. Most of the rest of the lab depends on the policy network working.
 - Implement [infrastructure/replay_buffer.py](lab4/infrastructure/replay_buffer.py) next so minibatches come back with aligned observations/actions/rewards/next observations/terminals.
 - Finish [infrastructure/utils.py](lab4/infrastructure/utils.py) so rollout collection and batching work before you debug training.
 - Then complete [policies/MLP_policy.py](lab4/policies/MLP_policy.py), especially `get_action` and `MLPPolicySL.update`.
 - Wire the pieces together in [agents/bc_agent.py](lab4/agents/bc_agent.py) and [infrastructure/rl_trainer.py](lab4/infrastructure/rl_trainer.py).

Coding notes:
 - Keep tensor and NumPy shapes consistent. A single observation should become a batch of size 1 before passing through the policy.
 - `policy.get_action(obs)` should return shape `(1, ac_dim)` for a single continuous-control observation.
 - In behavior cloning for continuous control, evaluation should use the predicted mean action directly rather than a noisy sampled action.
 - `ReplayBuffer.sample_random_data` must use the same random indices for every returned array so transitions stay aligned.
 - `sample_trajectories(...)` should keep collecting full rollouts until the batch contains at least the requested number of timesteps.
 - `sample_n_trajectories(...)` should return exactly `ntraj` rollouts.

Debugging tips:
 - Test one file at a time with short smoke runs instead of changing many TODOs before running anything.
 - Add temporary shape/value prints when debugging, especially around observations, actions, and sampled minibatches.
 - While debugging, use `--video_log_freq -1` to skip video rendering and speed up training.
 - For plain BC, `--n_iter 1` is the safest default. Larger `n_iter` values start collecting student rollouts, which is not the same as training only on expert demonstrations.
 - Check the latest `data/.../progress.csv` if training loss or evaluation return looks surprising.
 - If evaluation looks much worse than expected, sanity-check `get_action` before changing hyperparameters.

## Run the code

Tip: While debugging, you probably want to keep the flag `--video_log_freq -1` which will disable video logging and speed up the experiment. However, feel free to remove it to save videos of your awesome policy!

If running on Colab, adjust the `#@params` in the `Args` class according to the commmand line arguments above.

### Behavior Cloning

```
python lab4/scripts/run_lab4.py \
	--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
	--expert_data lab4/expert_data/expert_data_Ant-v2.pkl \
	--video_log_freq -1
```

What the main parameters mean:
 - `--env_name`: which control environment to train and evaluate in.
 - `--exp_name`: the label used when creating the run folder inside `data/`.
 - `--n_iter`: number of training iterations. For plain BC, start with `1`.
 - `--expert_data`: the demonstration dataset the policy imitates.
 - `--num_agent_train_steps_per_iter`: how many gradient updates to run on the current data.
 - `--train_batch_size`: minibatch size for each supervised update.
 - `--n_layers` and `--size`: the depth and width of the policy MLP.
 - `--learning_rate`: optimizer step size for behavior cloning.
 - `--video_log_freq`: how often to save rollout videos. Set `-1` while debugging to make runs faster.

What reward number to watch:
 - `Eval_AverageReturn` is the main policy reward metric. It is the average episode return of the current learned policy on evaluation rollouts.
 - `Initial_DataCollection_AverageReturn` is the average return of the expert demonstration data loaded at the beginning.
 - `Train_AverageReturn` is the average return of the trajectories added to the replay buffer that iteration. For `n_iter=1`, that initial batch is the expert data, not student-generated rollouts.

Make sure to also try another environment.
See the lab PDF for more details on what else you need to run.
To generate videos of the policy, remove the `--video_log_freq -1` flag.