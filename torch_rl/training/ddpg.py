import gym
from torch.optim import Adam
from torch_rl.utils import *
import time

from torch_rl.models import SimpleNetwork
from torch_rl.core import ActorCriticAgent
from torch_rl.envs import NormalisedActionsWrapper
from torch_rl.memory import HindsightMemory, SequentialMemory
from torch_rl.stats import RLTrainingStats
import copy

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


class Trainer(object):


    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()

    def train(self, num_episodes, max_episode_len, render=False, verbose=True, callbacks=[]):
        mvavg_reward = deque(maxlen=100)
        self._warmup()
        self.verbose = True
        for episode in range(num_episodes):
            self.state = self.env.reset()
            t_episode_start = time.time()
            acc_reward = 0
            self._episode_start()

            for step in range(max_episode_len):
                s, r, d, _ = self._episode_step(episode, acc_reward)
                if render:
                    self.env.render()
                if d:
                    break
                acc_reward += r

            #TODO implement these callbacks a bit better
            for callback in callbacks:
                if hasattr(callback, "episode_step"):
                    callback.episode_step(episode=episode, step=step, episode_reward=acc_reward)
            self._episode_end(episode)

            mvavg_reward.append(acc_reward)
            episode_time = time.time() - t_episode_start
            if verbose:
                prRed("#Training time: {:.2f} minutes".format(time.clock() / 60))
                prGreen("#Episode {}. Mvavg reward: {:.2f} Episode reward: {:.2f} Episode steps: {} Episode time: {:.2f} min"\
                        .format(episode, np.mean(mvavg_reward), acc_reward, step + 1, episode_time / 60))

    def _episode_step(self):
        raise NotImplementedError()

    def _episode_start(self):
        raise NotImplementedError()

    def _episode_end(self):
        raise NotImplementedError()

    def _warmup(self):
        pass


def reward_transform(env, obs, r, done, **kwargs):
    goal = kwargs.get('goal', None)
    if not goal is None:
        return -np.mean((obs - goal) ** 2)
    else:
        return r

class DDPGTrainer(Trainer):

    critic_criterion = mse_loss

    def __init__(self, env, actor, critic, num_episodes=2000, max_episode_len=500, batch_size=32, gamma=.99,
              replay_memory=SequentialMemory(1000000, window_length=1), tau=1e-3, lr_critic=1e-3, lr_actor=1e-4, warmup=2000, depsilon=1./5000,
                 epsilon=1., exploration_process=None,
                 optimizer_critic=None, optimizer_actor=None):
        super(DDPGTrainer, self).__init__(env)
        if exploration_process is None:
            self.random_process = OrnsteinUhlenbeckActionNoise(self.env.action_space.shape[0])
        else:
            self.random_process = exploration_process
        self.action_choice_function = random_process_action_choice(self.random_process)
        self.tau = tau
        self.lr_critic = lr_critic
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.replay_memory = replay_memory
        self.max_episode_len = max_episode_len
        self.epsilon = epsilon
        self.depsilon = depsilon
        self.warmup = warmup
        self.gamma = gamma
        self.target_critic = copy.deepcopy(critic)
        self.target_actor = copy.deepcopy(actor)
        self.optimizer_actor = Adam(actor.parameters(), lr=lr_actor) if optimizer_actor is None else optimizer_actor
        self.optimizer_critic = Adam(critic.parameters(), lr=lr_critic) if optimizer_critic is None else optimizer_critic

        self.goal_based = hasattr(env, "goal")

        self.target_agent = ActorCriticAgent(self.target_actor,self.target_critic)
        self.agent = ActorCriticAgent(actor, critic)

    def add_to_replay_memory(self,s,a,r,d):
        if self.goal_based:
            self.replay_memory.append(self.state, self.env.goal, a, r, d, training=True)
        else:
            self.replay_memory.append(self.state, a, r, d, training=True)

    def _warmup(self):

        for i in range(self.warmup):
            a = self.env.action_space.sample()
            s, r, d, _ = self.env.step(a)
            self.add_to_replay_memory(self.state, a, r, d)
            self.state = s

    def _episode_start(self):

        self.random_process.reset()

    def _episode_step(self, episode, acc_reward):
        if self.goal_based:
            action = self.agent.action(np.hstack((self.state, self.env.goal))).cpu().data.numpy()
        else:
            action = self.agent.action(self.state).cpu().data.numpy()

        # Choose action with exploration
        action = self.action_choice_function(action, self.epsilon)
        if self.epsilon > 0:
            self.epsilon -= self.depsilon

        state, reward, done, info = self.env.step(action)

        self.add_to_replay_memory(self.state, action, reward, done)
        self.state = state

        # Optimize over batch
        if self.goal_based:
            s1, g, a1, r, s2, terminal = self.replay_memory.sample_and_split(self.batch_size)
            s1 = np.hstack((s1,g))
            s2 = np.hstack((s2,g))
        else:
            s1, a1, r, s2, terminal = self.replay_memory.sample_and_split(self.batch_size)


        a2 = self.target_agent.actions(s2, volatile=True)

        q2 = self.target_agent.values(to_tensor(s2, volatile=True), a2, volatile=False)
        q2.volatile = False

        q_expected = to_tensor(np.asarray(r), volatile=False) + self.gamma * q2
        q_predicted = self.agent.values(to_tensor(s1), to_tensor(a1), requires_grad=True)

        self.optimizer_critic.zero_grad()
        critic_loss = DDPGTrainer.critic_criterion(q_expected, q_predicted)
        critic_loss.backward(retain_graph=True)
        self.optimizer_critic.step()
        # Actor optimization

        a1 = self.agent.actions(s1, requires_grad=True)
        q_input = tor.cat([to_tensor(s1), a1], 1)
        q = self.agent.values(q_input, requires_grad=True)
        loss_actor = -q.mean()

        self.optimizer_actor.zero_grad()
        loss_actor.backward(retain_graph=True)
        self.optimizer_actor.step()

        soft_update(self.target_agent.policy_network, self.agent.policy_network, self.tau)
        soft_update(self.target_agent.critic_network, self.agent.critic_network, self.tau)

        return state, reward, done, {}

    def _episode_end(self, episode):
        pass
