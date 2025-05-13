# dqn-pong



<br/>

# 1 - **Références du projet :**

* GitHub. (2024). *DQN-Atari-Pong* \[Code source]. [https://github.com/hrhouma/DQN-Atari-Pong](https://github.com/hrhouma/DQN-Atari-Pong)
* GitHub. (2024). *dqn-pong* \[Code source]. [https://github.com/hrhouma/dqn-pong](https://github.com/hrhouma/dqn-pong)





<br/>

# 2 - Description

### 1. Composants réutilisables et utilitaires

| Fichier                 | Description                                                                                                       |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `utils.py`              | Fonctions utilitaires : tracé des courbes d'apprentissage, création d'environnements, wrappers personnalisés Gym. |
| `preprocess_pseudocode` | Pseudocode détaillant les classes de prétraitement et de composition d'environnements.                            |
| `replay_memory.py`      | Implémentation du tampon de relecture (Replay Buffer) pour stocker et rejouer les transitions.                    |

---

### 2. Version PyTorch

| Fichier             | Description                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------- |
| `main_dqn.py`       | Script principal pour entraîner un agent DQN sur l’environnement `PongNoFrameskip-v4` avec PyTorch. |
| `dqn_agent.py`      | Agent DQN implémenté avec un réseau Q principal et un réseau Q cible.                               |
| `deep_q_network.py` | Architecture CNN du DQN (réseau de neurones convolutifs avec PyTorch).                              |

---

### 3. Version TensorFlow 2.x (dossier `tf2/`)

| Fichier                | Description                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------- |
| `tf2/agent.py`         | Agent DQN avec TensorFlow et Keras.                                                    |
| `tf2/main.py`          | Script principal d'entraînement avec TensorFlow 2.x.                                   |
| `tf2/network.py`       | Réseau de neurones convolutionnel avec Keras.                                          |
| `tf2/replay_memory.py` | Tampon de relecture adapté pour TensorFlow.                                            |
| `tf2/utils.py`         | Fonctions utilitaires TensorFlow, wrappers d’environnement, gestion de la mémoire GPU. |

---

### 4. Organisation suggérée des fichiers

```
.
├── main_dqn.py
├── dqn_agent.py
├── deep_q_network.py
├── replay_memory.py
├── utils.py
├── preprocess_pseudocode
├── tf2/
│   ├── agent.py
│   ├── main.py
│   ├── network.py
│   ├── replay_memory.py
│   └── utils.py
└── models/        ← Répertoire des checkpoints sauvegardés
```

---

### 5. Références GitHub associées

* GitHub. (2024). *DQN-Atari-Pong* \[Code source]. [https://github.com/hrhouma/DQN-Atari-Pong](https://github.com/hrhouma/DQN-Atari-Pong)
* GitHub. (2024). *dqn-pong* \[Code source]. [https://github.com/hrhouma/dqn-pong](https://github.com/hrhouma/dqn-pong)



<br/>

# 3 - Code
