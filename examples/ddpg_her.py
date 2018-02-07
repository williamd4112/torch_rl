import gym
from torch.optim import Adam
from torch_rl.utils import *
import time
import numpy as np

from torch_rl.models import SimpleNetwork
from torch_rl.core import ActorCriticAgent
from torch_rl.envs import NormalisedActionsWrapper
from torch_rl.memory import HindsightMemory, SequentialMemory
from torch_rl.stats import RLTrainingStats

"""
    Implementation of deep deterministic policy gradients with soft updates.

"""


def random_process_action_choice(random_process):
    def func(actor_output, epsilon):
        action = actor_output + epsilon * random_process()
        return action

    return func


def mse_loss(input, target):
    return tor.mean(tor.sum((input - target) ** 2))


# Set seed
np.random.seed(666)


# Training parameters
num_episodes = 80000
batch_size = 8
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
hindsight = True

replay_memory = HindsightMemory(limit=replay_capacity, window_length=1) if hindsight else SequentialMemory(replay_capacity, window_length=1)

env = NormalisedActionsWrapper(gym.make("Pendulum-v0"))
env.reset()
num_actions = env.action_space.shape[0]
num_observations = env.observation_space.shape[0]*2
relu, tanh = tor.nn.ReLU(), tor.nn.Tanh()

random_process = OrnsteinUhlenbeckActionNoise(num_actions, sigma=0.2)
action_choice_function = random_process_action_choice(random_process)

policy = cuda_if_available(SimpleNetwork([num_observations, middle_layer_size[0], middle_layer_size[1], num_actions],
                                         activation_functions=[relu, relu, tanh]))
target_policy = cuda_if_available(
    SimpleNetwork([num_observations, middle_layer_size[0], middle_layer_size[1], num_actions],
                  activation_functions=[relu, relu, tanh]))
critic = cuda_if_available(
    SimpleNetwork([num_observations + num_actions, middle_layer_size[0], middle_layer_size[1], 1],
                  activation_functions=[relu, relu]))
target_critic = cuda_if_available(
    SimpleNetwork([num_observations + num_actions, middle_layer_size[0], middle_layer_size[1], 1],
                  activation_functions=[relu, relu]))

policy.apply(gauss_init(0, weight_init_sigma))
critic.apply(gauss_init(0, weight_init_sigma))

hard_update(target_policy, policy)
hard_update(target_critic, critic)

target_agent = ActorCriticAgent(target_policy, target_critic)
agent = ActorCriticAgent(policy, critic)

optimizer_critic = Adam(agent.critic_network.parameters(), lr=critic_learning_rate, weight_decay=0)
optimizer_policy = Adam(agent.policy_network.parameters(), lr=actor_learning_rate, weight_decay=1e-2)
critic_criterion = mse_loss
stats = RLTrainingStats()

# Warmup phase
state = env.reset()
goal = env.observation_space.sample()
for i in loop_print("Warmup phase {}/{}", range(warmup)):
    action = env.action_space.sample()
    state_prev = state
    state, reward, done, info = env.step(action)
    replay_memory.append(state_prev, goal, action, reward, done)

env.reset()

def reward_transform(env, obs, r, done, **kwargs):
    goal = kwargs.get('goal', None)
    if not goal is None:
        return np.mean((obs-goal)**2)
    else:
        return r

for episode in range(num_episodes):

    state = env.reset()
    random_process.reset()
    acc_reward = 0
    t_episode_start = time.time()
    done = False
    goal = env.observation_space.sample()
    for i in range(max_episode_length):
        # env.render()
        action = agent.action(np.hstack((state, goal))).cpu().data.numpy()

        # Choose action with exploration
        action = action_choice_function(action, epsilon)
        if epsilon > 0:
            epsilon -= depsilon

        state_prev = state
        state, reward, done, info = env.step(action)
        reward = -1 if reward_transform(env, state, reward, done, goal=goal) > 1. else 1
        replay_memory.append(state_prev, goal, action, reward, done)

        acc_reward += reward
        # Optimize over batch

        s1, g, a1, r, s2, terminal = replay_memory.sample_and_split(batch_size)
        # Critic optimization
        s2 = np.hstack((s2,g))
        s1 = np.hstack((s1,g))
        a2 = target_agent.actions(s2, volatile=True)

        q2 = target_agent.values(to_tensor(s2, volatile=True), a2, volatile=False)
        q2.volatile = False

        q_expected = to_tensor(np.asarray(r), volatile=False) + gamma * q2
        q_predicted = agent.values(to_tensor(s1), to_tensor(a1), requires_grad=True)

        optimizer_critic.zero_grad()
        critic_loss = critic_criterion(q_expected, q_predicted)
        critic_loss.backward(retain_graph=True)
        optimizer_critic.step()
        # Actor optimization

        a1 = agent.actions(s1, requires_grad=True)
        q_input = tor.cat([to_tensor(s1), a1], 1)
        q = agent.values(q_input, requires_grad=True)
        loss_actor = -q.mean()

        optimizer_policy.zero_grad()
        loss_actor.backward(retain_graph=True)
        optimizer_policy.step()

        soft_update(target_agent.policy_network, agent.policy_network, tau)
        soft_update(target_agent.critic_network, agent.critic_network, tau)

        if done:
            break

    stats.episode_step(episode, acc_reward, epsilon=epsilon)

    episode_time = time.time() - t_episode_start
    prRed("#Training time: {:.2f} minutes".format(time.clock() / 60))
    prGreen("#Episode {}. Episode reward: {:.2f} Episode steps: {} Episode time: {:.2f} min".format(episode, acc_reward,
                                                                                                    i + 1,
                                                                                                    episode_time / 60))





