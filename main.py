# Let's define a grid world
import os
import numpy as np
from typing import List

from utils import visualise_policy, plot_world
from domain import KarelWorld, GymKarelWorld
from expertPolicy import ExpertPolicy

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


def play(env, model, render_dir: str = './env_renders', num_episodes: int = 5):
	
	# Enjoy trained agent
	success = 0
	for i in range(num_episodes):
		obs, info = env.reset()
		state_renders = [env.render(obs)]
		while True:
			action, _states = model.predict(obs, deterministic=True)
			obs, reward, done, truncated, info = env.step(action)
			state_renders.append(env.render(obs))
			if done or truncated:
				print("Reward:", reward)
				success = success + 1 if done and not truncated else success
				break
		
		# Let's save the images
		img_dir = os.path.join(render_dir, "episode_{}".format(i))
		os.makedirs(img_dir, exist_ok=True)
		visualise_policy(env.world, state_renders, img_dir)

	print("Success Rate: ", success / num_episodes)


def ppo_learn():
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
	
	logging_dir = f"logging/ppo_{run.id}"
	os.makedirs(logging_dir, exist_ok=True)
	
	model = PPO(config['policy_type'],
				env,
				verbose=1,
				tensorboard_log=logging_dir,
				policy_kwargs=config['policy_kwargs'],
				batch_size=64,
				ent_coef=0.0,
				learning_rate=3e-4,
				n_epochs=10,
				n_steps=64)
	model.learn(total_timesteps=config['total_timesteps'], callback=WandbCallback())
	
	model.save("ppo_karel")
	
	render_dir = os.path.join(logging_dir, "env_renders")
	os.makedirs(render_dir, exist_ok=True)
	play(env, model, render_dir)
	
	rng = np.random.default_rng()
	learner_rewards_after_training, _ = evaluate_policy(
		model, env, 100, return_episode_rewards=True
	)
	print(np.mean(learner_rewards_after_training))


def dqn_learn():
	
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
	
	render_dir = os.path.join(logging_dir, "env_renders")
	os.makedirs(render_dir, exist_ok=True)
	play(env, model, render_dir)
	
	rng = np.random.default_rng()
	learner_rewards_after_training, _ = evaluate_policy(
		model, env, 100, return_episode_rewards=True
	)
	print(np.mean(learner_rewards_after_training))


def expert():
	# Let's create a world and save the initial state as image
	world = KarelWorld((5, 5), env_type='random')
	state, info = world.reset()
	plot_world(world)
	
	# Let's get the expert actions
	expert_policy = ExpertPolicy(world)
	actions: List[int] = expert_policy.get_expert_actions(state)
	print("Actions: ", [world.action_space[i] for i in actions])

	# Let's visualise the actions
	state_renders = [world.render(state)]
	for action in actions:
		state, reward, done, info = world.step(action)
		state_renders.append(world.render(state))
		if done:
			break

	# Let's save the interaction as images and video
	visualise_policy(world, state_renders, './env_renders')
	

if __name__ == "__main__":
	# expert()
	dqn_learn()
	# ppo_learn()
	
	# from stable_baselines3.common.env_checker import check_env
	# check_env(GymKarelWorld((7, 7), env_type='fixed'))