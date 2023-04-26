# Let's define a grid world
import os
import numpy as np
from typing import List, Union

from utils import visualise_policy, plot_world
from domain import KarelWorld, GymKarelWorld, HERGymKarelWorld
from expertPolicy import ExpertPolicy
from time import time

from networks import CustomMLP
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from stable_baselines3 import PPO

from imitation.data import rollout
from imitation.algorithms.adversarial.gail import GAIL
from imitation.util.util import make_vec_env
from stable_baselines3.common.envs import BitFlippingEnv


def play(env: Union[KarelWorld, GymKarelWorld, HERGymKarelWorld],
		 model,
		 render_dir: str = './env_renders',
		 num_episodes: int = 5):
	
	if not os.path.exists(render_dir):
		os.makedirs(render_dir, exist_ok=True)
	
	# Enjoy trained agent
	success_rate = 0
	for i in range(num_episodes):
		obs, info = env.reset()
		# plot_world(env.world, img_name=os.path.join('./logging', "episode_{}.png".format(i)))
		
		state_renders = [env.render(obs)]
		actions, rewards = [], []
		while True:
			action, _states = model.predict(obs, deterministic=True)
			actions.append(env.action_mapping[int(action)])
			
			obs, reward, done, truncated, info = env.step(action)
			state_renders.append(env.render(obs))
			rewards.append(reward)
			
			if done or truncated:
				success_rate += info['is_success']
				break
		
		print("\n # ############################################### #")
		print("Episode: ", i, "\nActions: ", actions, "\nRewards: ", rewards)
		print("\nTotal Reward: ", sum(rewards), "\nSuccess: ", info['is_success'])
		
		# Let's save the images
		img_dir = os.path.join(render_dir, "episode_{}".format(i))
		os.makedirs(img_dir, exist_ok=True)
		visualise_policy(env.world, state_renders, img_dir)

	print("Test Success Rate: ", success_rate / num_episodes)


def random_play():
	logging_dir = "logging/dqn_play"
	os.makedirs(logging_dir, exist_ok=True)
	model = DQN.load("dqn_karel")
	render_dir = os.path.join(logging_dir, "env_renders")
	
	env = GymKarelWorld((5, 5), env_type='fixed_obs', max_steps=25)
	play(env, model, render_dir)


def ppo_learn():
	config = {
		"policy_type": "MlpPolicy",
		"total_timesteps": int(5e5),
		"policy_kwargs": dict(features_extractor_class=CustomMLP,
							  features_extractor_kwargs=dict(features_dim=256),
							  normalize_images=False),
		"world_size": 5,
		"env_type": 'fixed_obs',  # random, fixed_obs, fixed_all
		"max_steps": 25
	}
	
	env = GymKarelWorld((config['world_size'], config['world_size']),
						env_type=config['env_type'],
						max_steps=config['max_steps'])
	
	run = wandb.init(
		project="sb3",
		config=config,
		sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
		# monitor_gym=True,  # auto-upload the videos of agents playing the game
		# save_code=True,  # optional
	)
	
	logging_dir = f"logging/ppo_{run.id}"
	os.makedirs(logging_dir, exist_ok=True)
	
	model = PPO(config['policy_type'],
				env,
				verbose=1,
				tensorboard_log=logging_dir,
				policy_kwargs=config['policy_kwargs'],
				batch_size=64,  # set to n_steps
				ent_coef=0.0,
				learning_rate=3e-4,
				n_epochs=10,  # number of PPO epochs per rollout buffer
				n_steps=64  # rollout buffer size is n_steps * n_envs
				)
	model.learn(total_timesteps=config['total_timesteps'],
				progress_bar=True,
				callback=WandbCallback(model_save_path=os.path.join(logging_dir, "models"), verbose=2))
	
	model.save("ppo_karel")
	
	render_dir = os.path.join(logging_dir, "env_renders")
	play(env, model, render_dir)
	
	rng = np.random.default_rng()
	learner_rewards_after_training, _ = evaluate_policy(
		model, env, 100, return_episode_rewards=True
	)
	print(np.mean(learner_rewards_after_training))


def dqn_learn():
	
	ts = time()
	config = {
		"policy_type": "MlpPolicy",
		"total_timesteps": int(2e5),
		"policy_kwargs": dict(features_extractor_class=CustomMLP,
							  features_extractor_kwargs=dict(features_dim=256),
							  normalize_images=False),
		"world_size": 5,
		"env_type": 'fixed_obs',  # random, fixed_obs, fixed_all
		"max_steps": 25
	}
	
	env = GymKarelWorld((config['world_size'], config['world_size']),
						env_type=config['env_type'],
						max_steps=config['max_steps'])
	
	run = wandb.init(
		project="sb3",
		config=config,
		sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
		# monitor_gym=True,  # auto-upload the videos of agents playing the game
		# save_code=True,  # optional
	)

	logging_dir = f"logging/dqn_{run.id}"
	os.makedirs(logging_dir, exist_ok=True)
	
	model = DQN(config['policy_type'],
				env,
				policy_kwargs=config['policy_kwargs'],
				verbose=1,
				tensorboard_log=os.path.join(logging_dir, "runs"),
				# exploration_fraction=0.5,
				# exploration_initial_eps=1.0,
				# exploration_final_eps=0.1,
				)
	
	model.learn(total_timesteps=config['total_timesteps'],
				progress_bar=True,
				callback=WandbCallback(model_save_path=os.path.join(logging_dir, "models"), verbose=2))

	model.save("dqn_karel")
	print("Time taken: ", round(time() - ts, 3))
	
	render_dir = os.path.join(logging_dir, "env_renders")
	play(env, model, render_dir)
	
	rng = np.random.default_rng()
	learner_rewards_after_training, _ = evaluate_policy(
		model, env, 100, return_episode_rewards=True
	)
	print(np.mean(learner_rewards_after_training))


def expert_traj():
	
	# Let's create a world and save the initial state as image
	world = KarelWorld((5, 5), env_type='fixed_obs')
	state, info = world.reset()
	plot_world(world)
	
	# Let's get the expert actions
	expert_policy = ExpertPolicy(world)
	actions = expert_policy.get_expert_trajectory(state)
	print("Actions: ", [world.action_mapping[i] for i in actions])

	# Let's visualise the actions
	rewards = []
	state_renders = [world.render(state)]
	for action in actions:
		state, reward, done, info = world.step(action)
		rewards.append(reward)
		state_renders.append(world.render(state))
		if done:
			break

	# Reward per action
	print("Reward per action: ", rewards)
	# Total reward
	print("Total Reward: ", np.sum(rewards))

	# Let's save the interaction as images and video
	visualise_policy(world, state_renders, './env_renders')

	
def expert_step():
	config = {
		"policy_type": "MlpPolicy",
		"total_timesteps": int(2e5),
		"policy_kwargs": dict(features_extractor_class=CustomMLP,
							  features_extractor_kwargs=dict(features_dim=256),
							  normalize_images=False),
		"world_size": 5,
		"env_type": 'fixed_obs',  # random, fixed_obs, fixed_all
		"max_steps": 25
	}
	
	env = GymKarelWorld((config['world_size'], config['world_size']),
						env_type=config['env_type'],
						max_steps=config['max_steps'])
	
	model = ExpertPolicy(env.world)
	
	obs, info = env.reset()
	state_renders = [env.render(obs)]
	actions = []
	rewards = []
	success = 0
	while True:
		action, _states = model.predict(obs, deterministic=True)
		obs, reward, done, truncated, info = env.step(action)
		state_renders.append(env.render(obs))
		actions.append(int(action))
		rewards.append(reward)
		if done or truncated:
			success = success + 1 if done and not truncated else success
			break
			
	print("Actions: ", [env.action_mapping[i] for i in actions])
	print("Reward per action: ", rewards)
	print("Total Reward: ", np.sum(rewards))
	
	# Let's save the interaction as images and video
	visualise_policy(env.world, state_renders, './env_renders')
	

if __name__ == "__main__":
	# from stable_baselines3.common.env_checker import check_env
	# check_env(GymKarelWorld((7, 7), env_type='fixed'))
	
	# expert()
	# expert_step()
	# dqn_learn()
	# ppo_learn()
	# random_play()
