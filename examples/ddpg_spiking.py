import gym
from torch.optim import Adam
from torch_rl.utils import *
import time

from torch_rl.models import SimpleNetwork, Reservoir
from torch_rl.core import ActorCriticAgent
from torch_rl.envs import NormalisedActionsWrapper, NormalisedObservationsWrapper
from torch_rl.memory import SequentialMemory
from torch_rl.stats import RLTrainingStats

"""
    Implementation of deep deterministic policy gradients with soft updates.

"""

def random_process_action_choice(random_process):

    def func(actor_output, epsilon):
        action = actor_output + epsilon*random_process()
        return action
    return func

def mse_loss(input, target):
    return tor.mean(tor.sum((input-target)**2))

# Training parameters
param = Parameters()

param.num_episodes = 80000
param.batch_size = 128
param.tau = 0.001
param.epsilon = 1.0
param.depsilon = 1./50000
param.gamma = 0.99
param.replay_capacity = 1000000
param.warmup = 2000
param.max_episode_length = 500
param.actor_learning_rate = 1e-4
param.critic_learning_rate = 1e-3
param.middle_layer_size = [600,400,200]
param.weight_init_sigma = 0.003
param.reservoir_size = 200
param.recurrent_synapse_filter = 15e-4

replay_memory = SequentialMemory(limit=param.replay_capacity, window_length=1)


env = NormalisedObservationsWrapper(NormalisedActionsWrapper(gym.make("Pendulum-v0")))
env.reset()
num_actions = env.action_space.shape[0]
num_observations = env.observation_space.shape[0]
relu, tanh = tor.nn.ReLU(), tor.nn.Tanh()

random_process = OrnsteinUhlenbeckActionNoise(num_actions, sigma=0.2)
action_choice_function = random_process_action_choice(random_process)

policy = cuda_if_available(SimpleNetwork([param.reservoir_size, param.middle_layer_size[0], param.middle_layer_size[1], num_actions],
                           activation_functions=[relu,relu,tanh]))
target_policy = cuda_if_available(SimpleNetwork([param.reservoir_size, param.middle_layer_size[0], param.middle_layer_size[1], num_actions],
                           activation_functions=[relu,relu,tanh ]))
critic = cuda_if_available(SimpleNetwork([param.reservoir_size+num_actions, param.middle_layer_size[0], param.middle_layer_size[1], 1],
                           activation_functions=[relu,relu]))
target_critic =  cuda_if_available(SimpleNetwork([param.reservoir_size+num_actions,param.middle_layer_size[0], param.middle_layer_size[1], 1],
                           activation_functions=[relu,relu]))

policy.apply(gauss_init(0,param.weight_init_sigma))
critic.apply(gauss_init(0,param.weight_init_sigma))

hard_update(target_policy, policy)
hard_update(target_critic, critic)

target_agent = ActorCriticAgent(target_policy, target_critic)
agent = ActorCriticAgent(policy, critic)
spiking_net = Reservoir(0.1, 0.01,num_observations,param.reservoir_size, spectral_radius=0.9,
                        recursive=True, synapse_filter=15e-4)

optimizer_critic = Adam(agent.critic_network.parameters(), lr=param.critic_learning_rate, weight_decay=0)
optimizer_policy = Adam(agent.policy_network.parameters(), lr=param.actor_learning_rate, weight_decay=1e-2)
critic_criterion = mse_loss

stats = RLTrainingStats(save_rate=30, hyperparameters=param,
                        save_destination='/disk/no_backup/vlasteli/Projects/torch_rl/training_stats/ddpg_spiking_tanh_reservoir_params')


# Warmup phase
state = env.reset()
for i in loop_print("Warmup phase {}/{}",range(param.warmup)):
    action = env.action_space.sample()
    state_prev = spiking_net.forward(state).reshape(-1)
    state, reward, done, info = env.step(action)
    replay_memory.append(state_prev, action, reward, done)

env.reset()
for episode in range(param.num_episodes):

    state = env.reset()
    random_process.reset()
    acc_reward = 0
    t_episode_start = time.time()
    done = False
    spiking_net.reset()
    for i in range(param.max_episode_length):
        #env.render()

        state = spiking_net.forward(state).reshape(-1)

        action = agent.action(state).cpu().data.numpy()

        # Choose action with exploration
        action = action_choice_function(action, param.epsilon)
        param.epsilon-=param.depsilon

        state_prev = state
        state, reward, done, info = env.step(action)

        replay_memory.append(state_prev, action, reward, done)

        acc_reward += reward
        # Optimize over batch

        s1, a1, r, s2, terminal = replay_memory.sample_and_split(param.batch_size)
        # Critic optimization
        a2 = target_agent.actions(s2, volatile=True)

        q2 = target_agent.values(to_tensor(s2, volatile=True),a2, volatile=False)
        q2.volatile = False

        q_expected = to_tensor(np.asarray(r), volatile=False) + param.gamma*q2
        q_predicted = agent.values(to_tensor(s1), to_tensor(a1), requires_grad=True)


        optimizer_critic.zero_grad()
        critic_loss = critic_criterion(q_expected, q_predicted)
        critic_loss.backward(retain_graph=True)
        optimizer_critic.step()
        # Actor optimization

        a1 = agent.actions(s1, requires_grad=True)
        q_input = tor.cat([to_tensor(s1), a1],1)
        q = agent.values(q_input, requires_grad=True)
        loss_actor = -q.mean()
        
        optimizer_policy.zero_grad()
        loss_actor.backward(retain_graph=True)
        optimizer_policy.step()

        soft_update(target_agent.policy_network, agent.policy_network, param.tau)
        soft_update(target_agent.critic_network, agent.critic_network, param.tau)

        if done:
            break

    stats.episode_step(episode, acc_reward, loss_actor=loss_actor.cpu().data.numpy()[0],
                       loss_critic=critic_loss.cpu().data.numpy()[0])
    episode_time = time.time() - t_episode_start
    prRed("#Training time: {:.2f} minutes".format(time.clock()/60))
    prGreen("#Episode {}. Episode reward: {:.2f} Episode steps: {} Episode time: {:.2f} min".format(episode, acc_reward, i+1, episode_time/60))






