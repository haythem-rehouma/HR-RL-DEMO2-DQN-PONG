import gym
from gym import envs
env_names = [spec.id for spec in envs.registry.all()]
for name in sorted(env_names):
    if 'Pong' in name:
        print(name)