from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch.nn as nn
import torch as th


class CustomMLP(BaseFeaturesExtractor):
	"""
	    :param observation_space: (gym.Space)
	    :param features_dim: (int) Number of features extracted.
	        This corresponds to the number of unit for the last layer.
	"""
	def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
		super().__init__(observation_space, features_dim)
		# We assume HxWxC images (channels last)
		# Re-ordering will be done by pre-preprocessing or wrapper
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
		
	def forward(self, observations: th.Tensor) -> th.Tensor:
		return self.linear(self.mlp(observations))