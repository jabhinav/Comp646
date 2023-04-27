# Define the configuration variables
from networks import CustomMLP, CustomCNN

config = {
	"policy_type": "CnnPolicy",  # Supported: MlpPolicy, CnnPolicy
	"total_timesteps": int(1e5),  # 1e5 for 3x3 random/fixed, 3e5 for 5x5 fixed and 1e6 for 5x5 random
	"policy_kwargs": dict(features_extractor_class=CustomCNN,
						  features_extractor_kwargs=dict(features_dim=256),
						  normalize_images=False),
	"world_size": 3,  # Supported: 3, 5, 7
	"env_type": 'random',  # Supported: random, fixed, fixed_all[for debug, no learning]
	"max_steps": 10,  # 10 for 3x3, 25 for 5x5, 50 for 7x7
	"penalise_steps": True,  # Set to True for -0.05 per step
	"exploration_fraction": 0.1,
	"exploration_final_eps": 0.05,
	"exploration_initial_eps": 1.0,
	"learning_starts": 25000,  # 25000 for 3x3, 50000 for 5x5, 100000 for 7x7
}
