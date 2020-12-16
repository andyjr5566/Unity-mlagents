"""
solving pendulum using actor-critic model
"""

import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

import random
from collections import deque

def stack_samples(samples):
	array = np.array(samples)
	
	current_states = np.stack(array[:,0]).reshape((array.shape[0],-1))
	actions = np.stack(array[:,1]).reshape((array.shape[0],-1))
	rewards = np.stack(array[:,2]).reshape((array.shape[0],-1))
	new_states = np.stack(array[:,3]).reshape((array.shape[0],-1))
	dones = np.stack(array[:,4]).reshape((array.shape[0],-1))
	
	return current_states, actions, rewards, new_states, dones
	

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
	def __init__(self, state_dim, action_dim, sess):
		self.state_dim  = state_dim
		self.action_dim = action_dim
		self.sess = sess

		self.learning_rate = 0.0001
		self.epsilon = .9
		self.epsilon_decay = .99995
		self.gamma = .90
		self.tau   = .01

		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #

		self.memory = deque(maxlen=4000)
		self.actor_state_input, self.actor_model = self.create_actor_model()
		_, self.target_actor_model = self.create_actor_model()

		self.actor_critic_grad = tf.placeholder(tf.float32,
			[None, self.action_dim]) # where we will feed de/dC (from critic)

		actor_model_weights = self.actor_model.trainable_weights
		self.actor_grads = tf.gradients(self.actor_model.output,
			actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
		grads = zip(self.actor_grads, actor_model_weights)
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

		# ===================================================================== #
		#                              Critic Model                             #
		# ===================================================================== #

		self.critic_state_input, self.critic_action_input, \
			self.critic_model = self.create_critic_model()
		_, _, self.target_critic_model = self.create_critic_model()

		self.critic_grads = tf.gradients(self.critic_model.output,
			self.critic_action_input) # where we calcaulte de/dC for feeding above

		# Initialize for later gradient calculations
		self.sess.run(tf.initialize_all_variables())

	# ========================================================================= #
	#                              Model Definitions                            #
	# ========================================================================= #

	def create_actor_model(self):
		state_input = Input(shape=self.state_dim)
		h1 = Dense(500, activation='relu')(state_input)
		h2 = Dense(1000, activation='relu')(h1)
		h3 = Dense(500, activation='relu')(h2)
		output = Dense(self.action_dim, activation='tanh')(h3)

		model = Model(input=state_input, output=output)
		adam  = Adam(lr=0.0001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, model

	def create_critic_model(self):
		state_input = Input(shape=self.state_dim)
		state_h1 = Dense(500, activation='relu')(state_input)
		state_h2 = Dense(1000)(state_h1)

		action_input = Input(shape=self.action_dim,)
		action_h1    = Dense(500)(action_input)

		merged    = Concatenate()([state_h2, action_h1])
		merged_h1 = Dense(500, activation='relu')(merged)
		output = Dense(1, activation='linear')(merged_h1)
		model  = Model(input=[state_input,action_input], output=output)

		adam  = Adam(lr=0.0001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, action_input, model

	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #

	def remember(self, cur_state, action, reward, new_state, done):
		self.memory.append([cur_state, action, reward, new_state, done])

	def _train_actor(self, samples):
		
			cur_states, actions, rewards, new_states, _ =  stack_samples(samples)
			predicted_actions = self.actor_model.predict(cur_states)
			grads = self.sess.run(self.critic_grads, feed_dict={
				self.critic_state_input:  cur_states,
				self.critic_action_input: predicted_actions
			})[0]

			self.sess.run(self.optimize, feed_dict={
				self.actor_state_input: cur_states,
				self.actor_critic_grad: grads
			})

	def _train_critic(self, samples):
   

		cur_states, actions, rewards, new_states, dones = stack_samples(samples)
		target_actions = self.target_actor_model.predict(new_states)
		future_rewards = self.target_critic_model.predict([new_states, target_actions])
		
		rewards += self.gamma * future_rewards * (1 - dones)
		
		evaluation = self.critic_model.fit([cur_states, actions], rewards, verbose=0)
		#print(evaluation.history)
	def train(self):
		batch_size = 256
		if len(self.memory) < batch_size:
			return

		rewards = []
		samples = random.sample(self.memory, batch_size)
		self.samples = samples
		self._train_critic(samples)
		self._train_actor(samples)

	# ========================================================================= #
	#                         Target Model Updating                             #
	# ========================================================================= #

	def _update_actor_target(self):
		actor_model_weights  = self.actor_model.get_weights()
		actor_target_weights = self.target_actor_model.get_weights()
		
		for i in range(len(actor_target_weights)):
			actor_target_weights[i] = actor_model_weights[i]*self.tau + actor_target_weights[i]*(1-self.tau)
		self.target_actor_model.set_weights(actor_target_weights)

	def _update_critic_target(self):
		critic_model_weights  = self.critic_model.get_weights()
		critic_target_weights = self.target_critic_model.get_weights()
		
		for i in range(len(critic_target_weights)):
			critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
		self.target_critic_model.set_weights(critic_target_weights)

	def update_target(self):
		self._update_actor_target()
		self._update_critic_target()

	# ========================================================================= #
	#                              Model Predictions                            #
	# ========================================================================= #

	def act(self, cur_state):
		self.epsilon *= self.epsilon_decay
		if np.random.random() < self.epsilon:
			return self.actor_model.predict(cur_state)*2 + np.random.normal()
		return self.actor_model.predict(cur_state)*2