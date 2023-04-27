from typing import Union, Dict

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomMLP(BaseFeaturesExtractor):
	"""
	    :param observation_space: (gym.Space)
	    :param features_dim: (int) Number of features extracted.
	        This corresponds to the number of unit for the last layer.
	"""
	
	def __init__(self, observation_space: Union[spaces.Box, spaces.Dict], features_dim: int = 256):
		super().__init__(observation_space, features_dim)
		# We assume HxWxC images (channels last)
		# Re-ordering will be done by pre-preprocessing or wrapper
		if isinstance(observation_space, spaces.Dict):
			observation_space = observation_space["observation"]
		
		n_flattened_size = observation_space.shape[0] * observation_space.shape[1] * observation_space.shape[2]
		self.mlp = nn.Sequential(
			nn.Flatten(),
			nn.Linear(n_flattened_size, 512),
			nn.ReLU(),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, 256)
		)
		
		# Compute shape by doing one forward pass
		with th.no_grad():
			n_flatten = self.mlp(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
		
		self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
	
	def forward(self, observations: Union[Dict[str, th.Tensor], th.Tensor]) -> th.Tensor:
		if isinstance(observations, dict):
			observations = observations["observation"]
		return self.linear(self.mlp(observations))


class CustomCNN(BaseFeaturesExtractor):
	"""
	    :param observation_space: (gym.Space)
	    :param features_dim: (int) Number of features extracted.
	        This corresponds to the number of unit for the last layer.
	"""
	
	def __init__(self, observation_space: Union[spaces.Box, spaces.Dict], features_dim: int = 256):
		super().__init__(observation_space, features_dim)
		# We assume HxWxC images (channels last)
		# Re-ordering will be done by pre-preprocessing or wrapper
		if isinstance(observation_space, spaces.Dict):
			observation_space = observation_space["observation"]
		
		#  > single-layer CNN with 32 1 × 1 filters (no padding or stride)
		#  > LeakyReLU non-linearity
		#  > Flatten
		self.net = nn.Sequential(
			nn.Conv2d(4, 32, kernel_size=1),
			nn.LeakyReLU(),
			nn.Flatten(),
		)
		
		# Compute shape by doing one forward pass
		with th.no_grad():
			# Reshape the observation space to have channels first i.e. B x H x W x C -> B x C x H x W
			_observation = observation_space.sample()[None]
			_observation = th.as_tensor(_observation).float().permute(0, 3, 1, 2)
			n_flatten = self.net(_observation).shape[1]
			# n_flatten = self.net(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
		
		self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
	
	def forward(self, observations: Union[Dict[str, th.Tensor], th.Tensor]) -> th.Tensor:
		if isinstance(observations, dict):
			observations = observations["observation"]
		# Reshape the observation space to have channels first
		observations = observations.permute(0, 3, 1, 2)
		return self.linear(self.net(observations))
