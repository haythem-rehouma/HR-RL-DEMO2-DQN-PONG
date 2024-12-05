import gym
import numpy as np
from dqn_agent import DQNAgent
from utils import make_env
import time

# Création de l'environnement
env = make_env('PongNoFrameskip-v4')

# Configuration de l'agent avec les mêmes paramètres
agent = DQNAgent(gamma=0.99, 
                 epsilon=0.01,  # Epsilon bas car on utilise le modèle entraîné
                 lr=0.0001,
                 input_dims=(env.observation_space.shape),
                 n_actions=env.action_space.n, 
                 mem_size=50000, 
                 eps_min=0.1,
                 batch_size=32, 
                 replace=1000, 
                 eps_dec=1e-5,
                 chkpt_dir='models/', 
                 algo='DQNAgent',
                 env_name='PongNoFrameskip-v4')

# Chargement des modèles pré-entraînés
agent.load_models()

# Test du modèle
n_games = 5
for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.03)  # Pour voir le jeu plus lentement
        score += reward
        observation = observation_
    print(f'Partie {i+1} terminée avec un score de {score}')

env.close()