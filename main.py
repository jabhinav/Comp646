# Let's define a grid world
import os
from time import time

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback

import wandb
from domain import KarelWorld, GymKarelWorld
from expertPolicy import ExpertPolicy
from utils import visualise_policy, plot_world, play
from config import config


def random_DQN_play(model_path: str):
	"""
	Function to play a DQN agent in a random environment
	Args:
		model_path: Path to the saved model to be loaded (.zip file)

	"""
	
	logging_dir = "logging/dqn_play"
	os.makedirs(logging_dir, exist_ok=True)
	model = DQN.load(model_path)
	render_dir = os.path.join(logging_dir, "env_renders")
	
	env = GymKarelWorld((config['world_size'], config['world_size']),
							env_type=config['env_type'],
						max_steps=config['max_steps'])
	
	play(env, model, render_dir)


# # Because HER needs access to `env.compute_reward()`
# # HER must be loaded with the env HERGymKarelWorld
# model = DQN.load("her_dqn_karel", env=env)


def dqn_learn():
	"""
	Function to train DQN agent
	"""
	
	ts = time()
	
	env = GymKarelWorld((config['world_size'], config['world_size']),
						env_type=config['env_type'],
						max_steps=config['max_steps'],
						penalise_steps=config['penalise_steps'])
	
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
				exploration_fraction=config["exploration_fraction"],
				exploration_initial_eps=config["exploration_initial_eps"],
				exploration_final_eps=config["exploration_final_eps"],
				learning_starts=config["learning_starts"],
				)
	
	model.learn(
		total_timesteps=config['total_timesteps'], progress_bar=True,
		callback=WandbCallback(model_save_path=os.path.join(logging_dir, "models"), verbose=2),
	)
	
	model.save("dqn_karel")
	print("Time taken: ", round(time() - ts, 3))
	
	render_dir = os.path.join(logging_dir, "env_renders")
	play(env, model, render_dir)
	
	learner_rewards_after_training, _ = evaluate_policy(
		model, env, 100, return_episode_rewards=True
	)
	print(np.mean(learner_rewards_after_training))


def expert_traj():
	"""
	Function to generate expert trajectories by using initial state only
	"""
	
	# Let's create a world and save the initial state as image
	world = KarelWorld((config['world_size'], config['world_size']), env_type=config['env_type'])
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
	"""
	Function to generate expert trajectories by stepping through the environment
	Returns:

	"""
	
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
	dqn_learn()
# random_DQN_play()
