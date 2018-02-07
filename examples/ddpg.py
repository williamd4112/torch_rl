import gym
from torch_rl.utils import *

from torch_rl.models import SimpleNetwork
from torch_rl.envs import NormalisedActionsWrapper
from torch_rl.memory import SequentialMemory
from torch_rl.training import DDPGTrainer
from torch_rl.envs import SparseRewardGoalEnv
from torch_rl.stats import RLTrainingStats
import datetime
"""
    Implementation of deep deterministic policy gradients with soft updates.

"""

# Training parameters
num_episodes = 2000
batch_size = 64
tau = 0.001
epsilon = 1.0
depsilon = 1. / 50000
gamma = 0.99
replay_capacity = 1000000
warmup = 2000
max_episode_length = 500
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3
middle_layer_size = [400, 300]
weight_init_sigma = 0.003

replay_memory = SequentialMemory(limit=6000000, window_length=1)

env = SparseRewardGoalEnv(NormalisedActionsWrapper(gym.make("Pendulum-v0")))
env.reset()
num_actions = env.action_space.shape[0]
num_observations = env.observation_space.shape[0]*2
relu, tanh = tor.nn.ReLU(), tor.nn.Tanh()

actor = cuda_if_available(SimpleNetwork([num_observations, middle_layer_size[0], middle_layer_size[1], num_actions],
                                        activation_functions=[relu, relu, tanh]))

critic = cuda_if_available(
    SimpleNetwork([num_observations + num_actions, middle_layer_size[0], middle_layer_size[1], 1],
                  activation_functions=[relu, relu]))

actor.apply(gauss_init(0, weight_init_sigma))
critic.apply(gauss_init(0, weight_init_sigma))

# Training
trainer = DDPGTrainer(env=env, actor=actor, critic=critic,
                      tau=tau, epsilon=epsilon, batch_size=batch_size, depsilon=depsilon, gamma=gamma,
                      lr_actor=actor_learning_rate, lr_critic=critic_learning_rate, warmup=warmup, replay_memory=replay_memory
                      )

stats = RLTrainingStats(save_destination="/disk/no_backup/vlasteli/Projects/torch_rl/examples/ddpg_" + str(datetime.datetime.now()).replace(" ", "_"))

trainer.train(2000, max_episode_len=500, verbose=True, callbacks=[stats])
