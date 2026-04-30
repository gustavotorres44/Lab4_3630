import numpy as np
import time

############################################
############################################

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):

    # initialize env for the beginning of a new rollout
    ob, _ = env.reset()  # Gymnasium reset returns `(observation, info)`.

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render())
            if 'human' in render_mode:
                env.render()
                if hasattr(env, 'model'):
                    time.sleep(env.model.opt.timestep)

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob)  # `get_action` returns a batch; extract the single action below.
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        ob, rew, terminated, truncated, _ = env.step(ac)
        done = terminated or truncated

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # TODO: mark the final transition of the rollout.
        # HINT: end the rollout either when the environment is done or when
        # `steps` reaches `max_path_length`.
        raise NotImplementedError
        terminals.append(rollout_done)

        if rollout_done:
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect rollouts until the batch contains at least
        `min_timesteps_per_batch` environment steps.

        TODO: implement this function.
        HINT: call `sample_trajectory(...)` repeatedly, append each rollout to
        `paths`, and track the total number of collected timesteps with
        `get_pathlength(path)`.
    """
    timesteps_this_batch = 0
    paths = []
    raise NotImplementedError

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect ntraj rollouts.

        TODO: implement this function.
        HINT: call `sample_trajectory(...)` exactly `ntraj` times and return
        the resulting list of rollout dictionaries.
    """
    paths = []
    raise NotImplementedError

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])
