# dqn-pong



<br/>

# 1 - **Références du projet :**

* GitHub. (2024). *DQN-Atari-Pong* \[Code source]. [https://github.com/hrhouma/DQN-Atari-Pong](https://github.com/hrhouma/DQN-Atari-Pong)
* GitHub. (2024). *dqn-pong* \[Code source]. [https://github.com/hrhouma/dqn-pong](https://github.com/hrhouma/dqn-pong)





<br/>

# 2 - Description

### 2.1. Composants réutilisables et utilitaires

| Fichier                 | Description                                                                                                       |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `utils.py`              | Fonctions utilitaires : tracé des courbes d'apprentissage, création d'environnements, wrappers personnalisés Gym. |
| `preprocess_pseudocode` | Pseudocode détaillant les classes de prétraitement et de composition d'environnements.                            |
| `replay_memory.py`      | Implémentation du tampon de relecture (Replay Buffer) pour stocker et rejouer les transitions.                    |

---

### 2.2. Version PyTorch

| Fichier             | Description                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------- |
| `main_dqn.py`       | Script principal pour entraîner un agent DQN sur l’environnement `PongNoFrameskip-v4` avec PyTorch. |
| `dqn_agent.py`      | Agent DQN implémenté avec un réseau Q principal et un réseau Q cible.                               |
| `deep_q_network.py` | Architecture CNN du DQN (réseau de neurones convolutifs avec PyTorch).                              |

---

### 2.3. Version TensorFlow 2.x (dossier `tf2/`)

| Fichier                | Description                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------- |
| `tf2/agent.py`         | Agent DQN avec TensorFlow et Keras.                                                    |
| `tf2/main.py`          | Script principal d'entraînement avec TensorFlow 2.x.                                   |
| `tf2/network.py`       | Réseau de neurones convolutionnel avec Keras.                                          |
| `tf2/replay_memory.py` | Tampon de relecture adapté pour TensorFlow.                                            |
| `tf2/utils.py`         | Fonctions utilitaires TensorFlow, wrappers d’environnement, gestion de la mémoire GPU. |

---

### 2.4. Organisation suggérée des fichiers

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

### 2.5. Références GitHub associées

* GitHub. (2024). *DQN-Atari-Pong* \[Code source]. [https://github.com/hrhouma/DQN-Atari-Pong](https://github.com/hrhouma/DQN-Atari-Pong)
* GitHub. (2024). *dqn-pong* \[Code source]. [https://github.com/hrhouma/dqn-pong](https://github.com/hrhouma/dqn-pong)



<br/>

# 3 - Code



- https://github.com/hrhouma/DQN-Atari-Pong
- https://github.com/hrhouma/dqn-pong


# utils.py 

```python
import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gym

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0,
                 fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env.step(1)

        self.frame_buffer = np.zeros_like((2,self.shape))
        self.frame_buffer[0] = obs

        return obs

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=self.shape, dtype=np.float32)

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0

        return new_obs

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                            env.observation_space.low.repeat(repeat, axis=0),
                            env.observation_space.high.repeat(repeat, axis=0),
                            dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

def make_env(env_name, shape=(84,84,1), repeat=4, clip_rewards=False,
             no_ops=0, fire_first=False):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env
```


# replay_memory.py

```python
import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
```


# preprocess_pseudocode

```python
Class RepeatActionAndMaxFrame
  derives from: gym.Wrapper
  input: environment, repeat
  init frame buffer as an array of zeros in shape 2 x the obs space

  function step:
    input: action
  	set total reward to 0
    set done to false
  	for i in range repeat
  		call the env.step function
        receive obs, reward, done, info
  		increment total reward
  		insert obs in frame buffer
  		if done
            break
    end for
  	find the max frame
  	return: max frame, total reward, done, info

  function reset:
    input: none

  	call env.reset
  	reset the frame buffer
    store initial observation in buffer

    return: initial observation

Class PreprocessFrame
  derives from: gym.ObservationWrapper
  input: environment, new shape
  set shape by swapping channels axis
	set observation space to new shape using gym.spaces.Box (0 to 1.0)

	function observation
    input: raw observation
		covert the observation to gray scale
		resize observation to new shape
    convert observation to numpy array
    move observation's channel axis from position 2 to position 0
    observation /= 255
		return observation


Class StackFrames
  derives from: gym.ObservationWrapper
  input: environment, stack size
	init the new obs space (gym.spaces.Box) low & high bounds as repeat of n_steps
	initialize empty frame stack

	reset function
		clear the stack
		reset the environment
    for i in range(stack size)
   		append initial observation to stack
    convert stack to numpy array
    reshape stack array to observation space low shape
    return stack

	observation function
    input: observation
		append the observation to the end of the stack
		convert the stack to a numpy array
    reshape stack to observation space low shape
		return the stack of frames

function make_env:
  input: environment name, new shape, stack size
  init env with the base gym.make function
  env := RepeatActionAndMaxFrame
  env := PreprocessFrame
  env := StackFrames

  return: env
```


# main_dqn.py

```python
import gym
import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env
from gym import wrappers

if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    #env = gym.make('CartPole-v1')
    best_score = -np.inf
    load_checkpoint = False
    n_games = 250

    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                     batch_size=32, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='PongNoFrameskip-v4')

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    # if you want to record video of your agent playing, do a mkdir tmp && mkdir tmp/dqn-video
    # and uncomment the following 2 lines.
    #env = wrappers.Monitor(env, "tmp/dqn-video",
    #                    video_callable=lambda episode_id: True, force=True)
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                     reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
```


# dqn_agent.py

```python
import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
```



# deep_q_network.py

```python
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # conv3 shape is BS x n_filters x H x W
        conv_state = conv3.view(conv3.size()[0], -1)
        # conv_state shape is BS x (n_filters * H * W)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
```


# tf2/agent

```python
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from network import DeepQNetwork
from replay_memory import ReplayBuffer


class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.fname = self.chkpt_dir + self.env_name + '_' + self.algo + '_'

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(input_dims, n_actions)
        self.q_eval.compile(optimizer=Adam(learning_rate=lr))
        self.q_next = DeepQNetwork(input_dims, n_actions)
        self.q_next.compile(optimizer=Adam(learning_rate=lr))

    def save_models(self):
        self.q_eval.save(self.fname+'q_eval')
        self.q_next.save(self.fname+'q_next')
        print('... models saved successfully ...')

    def load_models(self):
        self.q_eval = keras.models.load_model(self.fname+'q_eval')
        self.q_next = keras.models.load_model(self.fname+'q_next')
        print('... models loaded successfully ...')

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                  self.memory.sample_buffer(self.batch_size)
        states = tf.convert_to_tensor(state)
        rewards = tf.convert_to_tensor(reward)
        dones = tf.convert_to_tensor(done)
        actions = tf.convert_to_tensor(action, dtype=tf.int32)
        states_ = tf.convert_to_tensor(new_state)
        return states, actions, rewards, states_, dones

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = tf.convert_to_tensor([observation])
            actions = self.q_eval(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        else:
            action = np.random.choice(self.action_space)
        return action

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = tf.range(self.batch_size, dtype=tf.int32)
        print(actions.shape)
        exit()
        action_indices = tf.stack([indices, actions], axis=1)

        with tf.GradientTape() as tape:
            q_pred = tf.gather_nd(self.q_eval(states), indices=action_indices)
            q_next = self.q_next(states_)

            max_actions = tf.math.argmax(q_next, axis=1, output_type=tf.int32)
            max_action_idx = tf.stack([indices, max_actions], axis=1)

            q_target = rewards + \
                self.gamma*tf.gather_nd(q_next, indices=max_action_idx) *\
                (1 - dones.numpy())

            loss = keras.losses.MSE(q_pred, q_target)

        params = self.q_eval.trainable_variables
        grads = tape.gradient(loss, params)

        self.q_eval.optimizer.apply_gradients(zip(grads, params))

        self.learn_step_counter += 1

        self.decrement_epsilon()
```

# tf2/main

```python
import numpy as np
from agent import Agent
from utils import plot_learning_curve, make_env, manage_memory
from gym import wrappers

if __name__ == '__main__':
    manage_memory()
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    load_checkpoint = False
    record_agent = False
    n_games = 250
    agent = Agent(gamma=0.99, epsilon=1, lr=0.0001,
                  input_dims=(env.observation_space.shape),
                  n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                  batch_size=32, replace=1000, eps_dec=1e-5,
                  chkpt_dir='models/', algo='DQNAgent',
                  env_name='PongNoFrameskip-v4')
    if load_checkpoint:
        agent.load_models()
        agent.epsilon = agent.eps_min

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' \
        + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    # if you want to record video of your agent playing, do a
    # mkdir video
    if record_agent:
        env = wrappers.Monitor(env, "video",
                               video_callable=lambda episode_id: True,
                               force=True)
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                       reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode {} score {:.1f} avg score {:.1f} '
              'best score {:.1f} epsilon {:.2f} steps {}'.
              format(i, score, avg_score, best_score, agent.epsilon,
                     n_steps))

        if score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = score

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
```

# tf2/network

```python
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten


class DeepQNetwork(keras.Model):
    def __init__(self, input_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.conv1 = Conv2D(32, 8, strides=(4, 4), activation='relu',
                            data_format='channels_first',
                            input_shape=input_dims)
        self.conv2 = Conv2D(64, 4, strides=(2, 2), activation='relu',
                            data_format='channels_first')
        self.conv3 = Conv2D(64, 3, strides=(1, 1), activation='relu',
                            data_format='channels_first')
        self.flat = Flatten()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(n_actions, activation=None)

    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
```

# tf2/replay_memory

```python
import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
```


# tf2/utils

```python
import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf


def manage_memory():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0,
                 fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env.step(1)

        self.frame_buffer = np.zeros_like((2,self.shape))
        self.frame_buffer[0] = obs

        return obs

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=self.shape, dtype=np.float32)

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0

        return new_obs

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                            env.observation_space.low.repeat(repeat, axis=0),
                            env.observation_space.high.repeat(repeat, axis=0),
                            dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

def make_env(env_name, shape=(84,84,1), repeat=4, clip_rewards=False,
             no_ops=0, fire_first=False):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env

```


















```python
import gym
import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env
from gym import wrappers

if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    load_checkpoint = False
    n_games = 250

    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                     batch_size=32, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='PongNoFrameskip-v4')

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    # Pour enregistrer une vidéo, créer d’abord le dossier tmp/dqn-video
    # env = wrappers.Monitor(env, "tmp/dqn-video",
    #                        video_callable=lambda episode_id: True, force=True)

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                       reward, observation_, done)
                agent.learn()

            observation = observation_
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode:', i, 'score:', score,
              ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
              'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
```


<br/>


# 4 -  Commandes partie 1

```python
cd .\dqn-pong\
tree ..\dqn-pong\ /f
```

```python
PS C:\Users\rehou\Documents\dqn-pong> tree ..\dqn-pong\ /f
Structure du dossier pour le volume OS
Le numéro de série du volume est 8433-41AA
C:\USERS\REHOU\DOCUMENTS\DQN-PONG
│   1.py.txt
│   2.py.txt
│   deep_q_network.py
│   dqn_agent.py
│   main_dqn.py
│   main_dqn_2.py
│   preprocess_pseudocode
│   README.md
│   replay_memory.py
│   test_env_1.py
│   test_env_2.py
│   test_env_3.py
│   utils.py
│
├───models
│       PongNoFrameskip-v4_DQNAgent_q_eval
│       PongNoFrameskip-v4_DQNAgent_q_next
│
├───plots
│       DQNAgent_PongNoFrameskip-v4_alpha0.0001_500games.png
│
├───tf2
│       agent.py
│       main.py
│       network.py
│       replay_memory.py
│       utils.py
│
└───__pycache__
        deep_q_network.cpython-36.pyc
        dqn_agent.cpython-36.pyc
        replay_memory.cpython-36.pyc
        utils.cpython-36.pyc
```



<br/>


# 5 -  Commandes partie 2


![image](https://github.com/user-attachments/assets/7185775a-2708-4dd1-a797-03c925978b9f)

![image](https://github.com/user-attachments/assets/1f4ec7e1-7601-4450-8013-154a48f58835)


```python
(env36) C:\Users\rehou> python --version
Python 3.6.13 :: Anaconda, Inc.
conda install numpy==1.16.6
conda install matplotlib==3.3.4 
conda install cloudpickle==1.6.0
conda install matplotlib==3.0.3
conda install scipy=1.5.3 -c conda-forge
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge opencv
conda list OpenCV
```


```python
import cv2
print(cv2.__version__)
exit()
pip install gym==0.21.0
# Gym 0.21.0 est la dernière version compatible avec Python 3.6_
# N'utilisez pas conda install gym car il y a des problèmes de compatibilité avec Python 3.6
pip install numpy==1.19.5
pip install cloudpickle==1.6.0
```

```python
import gym
print(gym.__version__)
exit()
```

```python
pip install gym[atari]
pip install autorom[accept-rom-license]
conda install -c conda-forge gym-atari
conda list
```

```python
import gym
print(gym.__version__)
exit()
```


```python
python 1.py.txt  
python test_env.py
python test_env_1.py  
python test_env_2.py  
python test_env_3.py 
python main_dqn.py  
python main_dqn_2.py  
```



<br/>


# 6 -  Commandes partie 3

- En cas de problèmes 

```python
pip uninstall torch torchvision torchaudio
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

# 
```python
python test_env_2.py
python main_dqn.py (SANS INTERFACE GRAPHIQUE)
python main_dqn_2.py (AVEC INTERFACE)
```




<br/>

# 7 - Structuration

- Le document `Dqn Pong Readme`  inclut :

* Une vue d'ensemble pédagogique du projet
* Une description claire et précise de chaque fichier
* La structure du dossier avec `tree`
* Toutes les commandes essentielles pour exécuter et configurer l’environnement


## 7.1 - Références du projet

* GitHub. (2024). *DQN-Atari-Pong* \[Code source]. [https://github.com/hrhouma/DQN-Atari-Pong](https://github.com/hrhouma/DQN-Atari-Pong)
* GitHub. (2024). *dqn-pong* \[Code source]. [https://github.com/hrhouma/dqn-pong](https://github.com/hrhouma/dqn-pong)

## 7.2 - Description des fichiers

### 7.2.1 Composants utilitaires communs

| Fichier                 | Description                                                                                                      |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `utils.py`              | Fonctions utilitaires : création d'environnements personnalisés Gym, wrappers, prétraitement, tracés matplotlib. |
| `replay_memory.py`      | Tampon de relecture (Replay Buffer) pour stocker les transitions de l'agent.                                     |
| `preprocess_pseudocode` | Pseudocode d'explication pour `RepeatActionAndMaxFrame`, `PreprocessFrame`, `StackFrames`.                       |

### 7.2.2 Version PyTorch

| Fichier             | Description                                                                             |
| ------------------- | --------------------------------------------------------------------------------------- |
| `main_dqn.py`       | Script principal d'entraînement avec PyTorch.                                           |
| `dqn_agent.py`      | Implémentation de l'agent DQN (choix d'action, apprentissage, réduction ε, sauvegarde). |
| `deep_q_network.py` | Réseau de neurones convolutionnel pour Q-Learning (CNN avec PyTorch).                   |

### 7.2.3 Version TensorFlow 2.x (dossier `tf2/`)

| Fichier                | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| `tf2/agent.py`         | Agent DQN avec réseau d'évaluation et réseau cible.              |
| `tf2/main.py`          | Script principal d'entraînement (TensorFlow).                    |
| `tf2/network.py`       | Réseau CNN avec Keras.                                           |
| `tf2/replay_memory.py` | Buffer adapté à TensorFlow.                                      |
| `tf2/utils.py`         | Même logique que `utils.py` avec gestion mémoire GPU TensorFlow. |

### 7.2.4 Structure du projet (dossier local)

```bash
C:\Users\rehou\Documents\dqn-pong
├── main_dqn.py
├── dqn_agent.py
├── deep_q_network.py
├── replay_memory.py
├── utils.py
├── preprocess_pseudocode
├── README.md
├── tf2/
│   ├── agent.py
│   ├── main.py
│   ├── network.py
│   ├── replay_memory.py
│   └── utils.py
├── models/               # Dossiers contenant les checkpoints
├── plots/                # Fichiers .png générés
└── __pycache__/          # Dossiers de cache Python
```

## 7.3 - Commandes utiles

### Afficher tous les fichiers et sous-dossiers du projet

```powershell
cd .\dqn-pong\
tree ..\dqn-pong\ /f
```

### Exemple de sortie :

```text
C:\USERS\REHOU\DOCUMENTS\DQN-PONG
│   deep_q_network.py
│   dqn_agent.py
│   main_dqn.py
│   replay_memory.py
│   utils.py
├───models
│       PongNoFrameskip-v4_DQNAgent_q_eval
│       PongNoFrameskip-v4_DQNAgent_q_next
├───plots
│       DQNAgent_PongNoFrameskip-v4_lr0.0001_250games.png
├───tf2
│       agent.py
│       main.py
│       network.py
│       replay_memory.py
│       utils.py
└───__pycache__
```

## 7.4 - Environnement Python 3.6 avec PyTorch 1.8.1 CPU

### Installation

```bash
conda install numpy==1.16.6
conda install matplotlib==3.3.4
conda install cloudpickle==1.6.0
conda install scipy=1.5.3 -c conda-forge
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge opencv
pip install gym==0.21.0
pip install gym[atari]
pip install autorom[accept-rom-license]
```

### Tests de version

```bash
python -c "import gym; print(gym.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

## 7.5 - Exécution des fichiers

```bash
python test_env_1.py
python test_env_2.py
python test_env_3.py
python main_dqn.py       # Entraînement PyTorch
python tf2/main.py       # Entraînement TensorFlow
```

### Nettoyage PyTorch en cas de conflit

```bash
pip uninstall torch torchvision torchaudio
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

## 7.6 - Liens GitHub

* [https://github.com/hrhouma/DQN-Atari-Pong](https://github.com/hrhouma/DQN-Atari-Pong)
* [https://github.com/hrhouma/dqn-pong](https://github.com/hrhouma/dqn-pong)


<br/>
<br/>


# 8 - Interprétations :

<br/> 

### 8.1. **Pourquoi l'agent obtient-il un score négatif ? Cela signifie-t-il qu’il "perd" ?**


```bash
... saving checkpoint ...
... saving checkpoint ...
episode:  1 score:  -20.0  average score -20.0 best score -20.00 epsilon 0.98 steps 1923
episode:  2 score:  -21.0  average score -20.3 best score -20.00 epsilon 0.97 steps 2687
episode:  3 score:  -21.0  average score -20.5 best score -20.00 epsilon 0.96 steps 3629
```

> RÉPONSE: 

- Oui, l'agent est en plein apprentissage ! Voici l'explication des différents éléments affichés :

## Indicateurs d'apprentissage

- **Score**: Les scores négatifs (-20.0, -21.0) sont normaux au début car l'agent perd contre l'adversaire
- **Epsilon**: Diminue progressivement (0.99 → 0.98 → 0.97) ce qui indique que l'agent explore de moins en moins et exploite de plus en plus sa connaissance
- **Steps**: Augmente (932 → 1923 → 2687) montrant que l'agent accumule de l'expérience

## Processus d'apprentissage

L'agent utilise l'algorithme DQN (Deep Q-Network) avec :
- Exploration aléatoire au début (epsilon élevé)
- Stockage des expériences dans la mémoire replay
- Apprentissage progressif du réseau de neurones
- Sauvegarde des meilleurs modèles ("saving checkpoint")

## Points importants

- L'avertissement "VisibleDeprecationWarning" n'affecte pas l'apprentissage
- Les scores négatifs sont normaux dans les premiers épisodes
- L'agent a besoin de plusieurs milliers d'épisodes pour commencer à s'améliorer
- Le "best score" s'actualisera quand l'agent commencera à s'améliorer

L'apprentissage est un processus long qui nécessite beaucoup d'itérations avant de voir des améliorations significatives dans les performances de l'agent.






<br/> 

### 8.2. Structure du Projet et explications supplémentaires

**Architecture principale**
- L'agent DQN (Deep Q-Network) apprend à jouer à Pong
- L'environnement est prétraité pour optimiser l'apprentissage
- Le modèle utilise PyTorch pour l'apprentissage profond

## Composants Principaux

**Environnement de jeu**
- Utilise 'PongNoFrameskip-v4' de OpenAI Gym
- Les images sont :
  - Converties en niveaux de gris
  - Redimensionnées à 84x84 pixels
  - Normalisées entre 0 et 1
  - Empilées par 4 pour capturer le mouvement

**Agent DQN**
- Epsilon-greedy pour l'exploration (commence à 1.0)
- Diminue progressivement jusqu'à 0.1
- Mémoire replay de 50000 transitions
- Taux d'apprentissage de 0.0001
- Facteur d'actualisation (gamma) de 0.99

## Processus d'Apprentissage

**Boucle principale**
1. L'agent observe l'état du jeu
2. Choisit une action (exploration ou exploitation)
3. Exécute l'action et obtient une récompense
4. Stocke l'expérience dans la mémoire replay
5. Apprend périodiquement sur un batch de 32 expériences

**Réseau de neurones**
- 3 couches convolutives
- 2 couches entièrement connectées
- Optimiseur RMSprop
- Fonction de perte MSE

## Indicateurs visibles

- Score : points marqués dans la partie
- Average score : moyenne mobile sur 100 épisodes
- Epsilon : taux d'exploration actuel
- Steps : nombre total d'actions effectuées

## Interface graphique

La fenêtre affichée montre :
- La barre verte : la raquette de l'agent
- La barre brune : l'adversaire
- Le carré vert : la balle
- Le fond marron : le terrain de jeu

L'apprentissage prend du temps et les scores négatifs au début sont normaux car l'agent apprend progressivement à jouer.

<br/> 

## 8.3. Conclusion

Dans ce jeu de Pong, voici les joueurs :

## L'agent DQN (Deep Q-Network)
- Contrôle la raquette verte à droite
- C'est l'agent qui apprend à jouer via l'apprentissage par renforcement
- Utilise un réseau de neurones profond pour apprendre la stratégie optimale

## L'adversaire
- La raquette marron à gauche
- C'est un adversaire intégré au jeu (IA simple)
- Fait partie de l'environnement Pong d'Atari

## Fonctionnement
- L'agent DQN observe l'état du jeu (positions des raquettes et de la balle)
- Il choisit une action (monter ou descendre sa raquette)
- Il reçoit une récompense (+1 quand il marque, -1 quand il encaisse un point)
- Il apprend progressivement à jouer en optimisant ses actions pour maximiser ses récompenses

Les scores négatifs au début (-20, -21) indiquent que l'agent DQN (raquette verte) est en train d'apprendre et perd contre l'adversaire intégré (raquette marron).



## Les épisodes
- Un épisode est une partie complète de Pong qui se termine quand un joueur atteint 21 points
- Le score -21.0 signifie que l'agent a perdu la partie en encaissant 21 points

## Les métriques

**Score**
- Score négatif (-20.0, -21.0) indique le nombre de points encaissés par l'agent
- Plus le score est proche de 0, meilleur est l'agent

**Average score**
- Moyenne mobile des 100 derniers scores
- Passe de -20.0 à -20.7, montrant une légère dégradation des performances

**Epsilon (ε)**
- Commence à 0.98 et diminue progressivement (0.98 → 0.97 → 0.96)
- Représente la probabilité que l'agent fasse une action aléatoire
- Plus epsilon diminue, plus l'agent utilise sa stratégie apprise plutôt que l'exploration aléatoire

**Steps**
- Nombre total d'actions effectuées depuis le début de l'entraînement
- Augmente de 1923 → 2687 → 3629 → 4481 → 5481
- N'indique pas directement une amélioration, mais plutôt l'expérience accumulée par l'agent

## Progression
- L'agent est encore en phase d'apprentissage initial
- Les scores négatifs constants montrent qu'il n'a pas encore appris de stratégie efficace
- La diminution d'epsilon indique qu'il commence à moins explorer et plus exploiter sa connaissance











# 9 - Annexe - troubleshooting


- *Pour vérifier les versions installées de torch, torchvision et torchaudio dans votre environnement conda env36, utilisez les commandes suivantes :*

```bash
conda list torch
conda list torchvision
conda list torchaudio
```

- Si ces packages ne sont pas installés via conda, vous pouvez vérifier avec pip :

```bash
pip list | grep torch
```

- Si les packages ne sont pas installés, vous verrez une sortie vide. Dans ce cas, vous devrez les installer en utilisant une version compatible avec Python 3.6. Voici la commande recommandée pour installer ces packages :

```bash
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

- Cette commande installera les versions CPU de PyTorch, torchvision et torchaudio compatibles avec Python 3.6.















