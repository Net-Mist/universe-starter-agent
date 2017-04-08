import gym
from gym import wrappers


def create_env(env_id, record=False, **kwargs):
    spec = gym.spec(env_id)

    if spec.tags.get('flashgames', False):
        return
    elif spec.tags.get('atari', False) and spec.tags.get('vnc', False):
        return
    else:
        # Assume atari.
        assert "." not in env_id  # universe environments have dots in names.
        return create_atari_env(env_id, record)


def create_atari_env(env_id, record=False):
    env = gym.make(env_id)
    if record:
        env = wrappers.Monitor(env, '/tmp/experiment')
    return env
