import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import pdb
from deeprm.DeepRMEnv import DeepRMEnv  
from deeprm import parameters
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_CNN = False
# use_CNN = True


class Memory:
	def __init__(self):
		self.actions = []
		self.states = []
		self.logprobs = []
		self.rewards = []
		self.is_terminals = []
	
	def clear_memory(self):
		del self.actions[:]
		del self.states[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.is_terminals[:]

class ActorCritic(nn.Module):
	def __init__(self, state_dim, action_dim, n_latent_var):
		super(ActorCritic, self).__init__()

		# actor
		self.action_layer = nn.Sequential(
				nn.Linear(state_dim, n_latent_var),
				nn.Tanh(),
				nn.Linear(n_latent_var, n_latent_var),
				nn.Tanh(),
				nn.Linear(n_latent_var, action_dim),
				nn.Softmax(dim=-1)
				)
		self.CNN_action_layer = nn.Sequential(
			nn.Conv2d(1,8,3),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),
			nn.Conv2d(8,4,3)
			)
		self.CNN_action_layer_2 = nn.Sequential(
				nn.Tanh(),
				nn.Linear(3052, action_dim),
				nn.Softmax(dim=-1)
				)
		# critic
		self.value_layer = nn.Sequential(
				nn.Linear(state_dim, n_latent_var),
				nn.Tanh(),
				nn.Linear(n_latent_var, n_latent_var),
				nn.Tanh(),
				nn.Linear(n_latent_var, 1)
				)

		self.CNN_value_layer = nn.Sequential(
			nn.Conv2d(1,8,3),
			nn.ReLU(),
			nn.MaxPool2d(2, stride=2),
			nn.Conv2d(8,4,3)
			)
		self.CNN_value_layer_2 = nn.Sequential(
				nn.Tanh(),
				nn.Linear(3052, n_latent_var),
				nn.Tanh(),
				nn.Linear(n_latent_var, 1)
				)
		
	def forward(self):
		raise NotImplementedError
		
	def act(self, state, memory):
		state = torch.from_numpy(state).float().to(device) 
		if use_CNN:
			inter_value = self.CNN_action_layer(state)
			action_probs = self.CNN_action_layer_2(inter_value.reshape(inter_value.shape[0], -1))
		else:
			action_probs = self.action_layer(state)
		dist = Categorical(action_probs)
		action = dist.sample()
		
		memory.states.append(state)
		memory.actions.append(action)
		memory.logprobs.append(dist.log_prob(action))
		
		return action.item()
	
	def evaluate(self, state, action):
		if use_CNN:
			inter_value = self.CNN_action_layer(state)
			action_probs = self.CNN_action_layer_2(inter_value.reshape(inter_value.shape[0], -1))
		else:
			action_probs = self.action_layer(state)
		dist = Categorical(action_probs)
		
		action_logprobs = dist.log_prob(action)
		dist_entropy = dist.entropy()
		
		if use_CNN:
			inter_value = self.CNN_value_layer(state)
			state_value = self.CNN_value_layer_2(inter_value.reshape(inter_value.shape[0], -1))
		else:
			state_value = self.value_layer(state)
		
		return action_logprobs, torch.squeeze(state_value), dist_entropy
		
class PPO:
	def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
		self.lr = lr
		self.betas = betas
		self.gamma = gamma
		self.eps_clip = eps_clip
		self.K_epochs = K_epochs
		
		self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
		self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
		self.policy_old.load_state_dict(self.policy.state_dict())
		
		self.MseLoss = nn.MSELoss()
	
	def update(self, memory):   
		# Monte Carlo estimate of state rewards:
		rewards = []
		discounted_reward = 0
		for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			rewards.insert(0, discounted_reward)
		
		# Normalizing the rewards:
		rewards = torch.tensor(rewards).to(device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
		
		# convert list to tensor
		old_states = torch.stack(memory.states).to(device).detach()
		old_actions = torch.stack(memory.actions).to(device).detach()
		old_logprobs = torch.stack(memory.logprobs).to(device).detach()
		# Optimize policy for K epochs:
		if use_CNN:
			old_states = old_states.squeeze(1)
		for _ in range(self.K_epochs):
			# Evaluating old actions and values :
			logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
			
			# Finding the ratio (pi_theta / pi_theta__old):
			ratios = torch.exp(logprobs - old_logprobs.detach())
				
			# Finding Surrogate Loss:
			advantages = rewards - state_values.detach()
			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
			loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
			
			# take gradient step
			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()
		
		# Copy new weights into old policy:
		self.policy_old.load_state_dict(self.policy.state_dict())
		
def main():
	############## Hyperparameters ##############
	pa = parameters.Parameters()

	pa.simu_len = 50  # 1000
	pa.num_ex = 1000 # 100
	# pa.num_ex = 1
	pa.num_nw = 10
	pa.new_job_rate = 0.1
	pa.episode_max_length = 2000  # 2000

	pa.compute_dependent_parameters()
	# env = DeepRMEnv(pa, play=False, repre='image', test_mode=True, log_path="./logs")
	env = DeepRMEnv(pa, play=False, repre='image', test_mode=False, log_path="./logs")

	state_dim = 20*223
	action_dim = 10
	render = False
	solved_reward = 10         # stop training if avg_reward > solved_reward
	log_interval = 20           # print avg reward in the interval
	max_episodes = 50000       # max training episodes
	max_timesteps = 2000         # max timesteps in one episode
	n_latent_var = 256           # number of variables in hidden layer
	update_timestep = 2000      # update policy every n timesteps
	lr = 0.002
	betas = (0.9, 0.999)
	gamma = 0.99                # discount factor
	K_epochs = 4                # update policy for K epochs
	eps_clip = 0.2              # clip parameter for PPO
	random_seed = 0
	#############################################
	
	if random_seed:
		torch.manual_seed(random_seed)
		env.seed(random_seed)
	
	memory = Memory()
	ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
	print(lr,betas)
	
	# logging variables
	running_reward = 0
	avg_length = 0
	timestep = 0
	total_timestep = 0
	
	# training loop
	for i_episode in range(1, max_episodes):
		if i_episode % (max_episodes//5) == 0:
			ii = i_episode // (max_episodes//5)
			pa.new_job_rate = ii * 0.1
			env = DeepRMEnv(pa, play=False, repre='image', test_mode=False, log_path="./logs")
			state = env.reset()
		else:
			state = env.reset()
		if use_CNN:
			state = np.expand_dims(np.expand_dims(state, axis=0), 0)
		else:
			state = state.reshape(-1)
		for t in range(max_timesteps):
			timestep += 1
			total_timestep += 1
			
			# Running policy_old:
			action = ppo.policy_old.act(state, memory)
			state, reward, done, _ = env.step(action)
			if use_CNN:
				state = np.expand_dims(np.expand_dims(state, axis=0), 0)
			else:
				state = state.reshape(-1)
			# Saving reward and is_terminal:
			memory.rewards.append(reward)
			memory.is_terminals.append(done)
			
			# update if its time
			if timestep % update_timestep == 0:
				ppo.update(memory)
				memory.clear_memory()
				timestep = 0
			
			running_reward += reward
			if render:
				env.render()
			if done:
				break

		avg_length += t
		
		# stop training if avg_reward > solved_reward
		if running_reward > (log_interval*solved_reward):
			print("########## Solved! ##########")
			torch.save(ppo.policy.state_dict(), './PPO_RM_{}.pth'.format(pa.new_job_rate))
			break
			
		# logging
		if i_episode % log_interval == 0:
			avg_length = int(avg_length/log_interval)
			running_reward = int((running_reward/log_interval))
			
			print('Episode {} \t avg length: {} \t reward: {} timestep: {}'.format(i_episode, avg_length, running_reward, total_timestep))
			running_reward = 0
			avg_length = 0
	
	# Test 
	pa.unseen = True
	pa.num_ex = 500 # 100
	pa.new_job_rate = 0.4
	test_reward = []
	mean_discounted_reward = []
	env = DeepRMEnv(pa, play=False, repre='image', test_mode=True, log_path="./logs")
	state = env.reset()
	for i in range(100000):
		state = state.reshape(-1)
		action = ppo.policy_old.act(state, memory)
		state, reward, done, _ = env.step(action)
		test_reward.append(reward)
		
		if done:
			discounted_reward = 0
			for r in reversed(test_reward):
				discounted_reward = r + (gamma * discounted_reward)
			mean_discounted_reward.append(discounted_reward)
		if i % 1000 == 0:
			print('Discounted reward: {:.4f}'.format(np.mean(mean_discounted_reward)))
			test_reward = []

if __name__ == '__main__':
	main()
	
