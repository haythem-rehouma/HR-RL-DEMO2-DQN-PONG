import gym
import numpy as np
from utils import make_env

# Création de l'environnement
env = make_env('PongNoFrameskip-v4')

# Paramètres
n_games = 5  # Nombre de parties à jouer
render = True  # Afficher le jeu

# Boucle principale
for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    
    while not done:
        # Action aléatoire
        action = env.action_space.sample()
        
        # Exécuter l'action
        observation_, reward, done, info = env.step(action)
        
        # Mettre à jour le score
        score += reward
        
        # Mettre à jour l'observation
        observation = observation_
        
        if render:
            env.render()
    
    print(f'Episode: {i}, Score: {score}')

env.close()