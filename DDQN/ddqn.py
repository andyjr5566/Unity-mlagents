import sys
import random
import numpy as np

from tqdm import tqdm
from .agent import Agent
from random import random, randrange

from utils.memory_buffer import MemoryBuffer
from utils.networks import tfSummary
from utils.stats import gather_stats

class DDQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, 
                action_dim, 
                state_dim, 
                with_per = True, 
                dueling = True, 
                nb_episodes = 100000,
                batch_size = 32,
                gather_stats = True,
                epsilon = 0.9,
                epsilon_decay = 0.99999,
                lr = 2.5e-4
                ):
        """ Initialization
        """
        # Environment and DDQN parameters
        self.with_per = with_per
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gather_stats = gather_stats
        #
        self.lr = lr
        self.gamma = 0.95
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.buffer_size = 20000
        self.nb_episodes = nb_episodes
        self.batch_size = batch_size

        #
        if(len(state_dim) < 3):
            self.tau = 1e-2
        else:
            self.tau = 1.0
        # Create actor and critic networks
        self.agent = Agent(self.state_dim, self.action_dim, self.lr, self.tau, dueling)
        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(self.buffer_size, with_per)

    def policy_action(self, s):
        """ Apply an espilon-greedy policy to pick next action
        """
        return np.argmax(self.agent.predict(s),axis=1)

    def train_agent(self, batch_size):
        """ Train Q-network on batch sampled from the buffer
        """
        # Sample experience from memory buffer (optionally with PER)
        s, a, r, d, new_s, idx = self.buffer.sample_batch(batch_size)

        # Apply Bellman Equation on batch samples to train our DDQN
        q = self.agent.predict(s)
        next_q = self.agent.predict(new_s)
        q_targ = self.agent.target_predict(new_s)

        for i in range(s.shape[0]):
            old_q = q[i, a[i]]
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = np.argmax(next_q[i,:])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]
            if(self.with_per):
                # Update PER Sum Tree
                self.buffer.update(idx[i], abs(old_q - q[i, a[i]]))
        # Train on batch
        self.agent.fit(s, q)
        # Decay epsilon
        self.epsilon *= self.epsilon_decay


    def train(self, env, summary_writer):
        """ Main DDQN Training Algorithm
        """

        results = []
        tqdm_e = tqdm(range(self.nb_episodes), desc='Score', leave=True, unit=" episodes")

        for e in tqdm_e:
            # Reset episode
            time, cumul_reward, done  = 0, 0, False
            old_state = env.reset()

            while not done:
                
                # Actor picks an action (following the policy)
                a = self.policy_action(old_state)
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a)
                # Memorize for experience replay
                self.memorize(old_state, a, r, done, new_state)
                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1
                # Train DDQN and transfer weights to target network
                if(self.buffer.size() > self.batch_size):
                    self.train_agent(self.batch_size)
                    self.agent.transfer_weights()

            # Gather stats every episode for plotting
            if(self.gather_stats):
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()

            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()

        return results

    def memorize(self, states, actions, rewards, dones, new_states):
        """ Store experience in memory buffer
        """
        for state, action, reward, done, new_state in zip(states, actions, rewards, dones, new_states):
            state = state[np.newaxis,:]
            new_state = new_state[np.newaxis,:]

            if(self.with_per):
                q_val = self.agent.predict(state)
                q_val_t = self.agent.target_predict(new_state)
                next_best_action = np.argmax(self.agent.predict(new_state))
                new_val = reward + self.gamma * q_val_t[0, next_best_action]
                td_error = abs(new_val - q_val)[0]
            else:
                td_error = 0
            self.buffer.memorize(state, action, reward, done, new_state, td_error)

    def save(self, path):
        # path += '_LR_{}'.format(self.lr)
        # if(self.with_per):
        #     path += '_PER'
        self.agent.save(path)

    def load_weights(self, path):
        # path += '_LR_{}'.format(self.lr)
        # if(self.with_per):
        #     path += '_PER'
        self.agent.load_weights(path)
