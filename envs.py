import gym


def create_env(env_id, client_id, remotes, **kwargs):
    spec = gym.spec(env_id)

    if spec.tags.get('flashgames', False):
        return
    elif spec.tags.get('atari', False) and spec.tags.get('vnc', False):
        return
    else:
        # Assume atari.
        assert "." not in env_id  # universe environments have dots in names.
        return create_atari_env(env_id)


def create_atari_env(env_id):
    env = gym.make(env_id)
    return env
