import os
from time import time

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback

import wandb
from domain import HERGymKarelWorld, GymKarelWorld
from utils import play
from config import config



def dqn_learn_with_her():
	from stable_baselines3 import HerReplayBuffer
	
	ts = time()
	# Add HER + DQN to the dict
	config['replay_buffer_kwargs'] = dict(n_sampled_goal=4,
										  # Number of virtual transitions to create per real transition
										  goal_selection_strategy="future",
										  # Available strategies (cf paper): future, final, episode
										  )
	env = HERGymKarelWorld((config['world_size'], config['world_size']),
						   env_type=config['env_type'],
						   max_steps=config['max_steps'])
	
	run = wandb.init(
		project="sb3",
		config=config,
		sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
		# monitor_gym=True,  # auto-upload the videos of agents playing the game
		# save_code=True,  # optional
	)
	
	logging_dir = f"logging/her_dqn_{run.id}"
	os.makedirs(logging_dir, exist_ok=True)
	
	model = DQN(
		"MultiInputPolicy",
		env,
		replay_buffer_class=HerReplayBuffer,
		policy_kwargs=config['policy_kwargs'],
		# Parameters for HER
		replay_buffer_kwargs=config['replay_buffer_kwargs'],
		verbose=1,
		tensorboard_log=os.path.join(logging_dir, "runs"),
		# exploration_fraction=0.5,
		# exploration_initial_eps=1.0,
		# exploration_final_eps=0.1,
		# learning_starts=2000,
	)
	
	model.learn(total_timesteps=config['total_timesteps'],
				progress_bar=True,
				callback=WandbCallback(model_save_path=os.path.join(logging_dir, "models"), verbose=2))
	
	model.learn(total_timesteps=config['total_timesteps'], progress_bar=True)
	
	model.save("her_dqn_karel")
	
	print("Time taken: ", round(time() - ts, 3))
	
	render_dir = os.path.join(logging_dir, "env_renders")
	play(env, model, render_dir)
	
	learner_rewards_after_training, _ = evaluate_policy(
		model, env, 100, return_episode_rewards=True
	)
	print(np.mean(learner_rewards_after_training))


def ppo_learn():
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
