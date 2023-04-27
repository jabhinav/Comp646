# Define the configuration variables
from networks import CustomMLP, CustomCNN

config = {
	"policy_type": "MlpPolicy",  # Supported: MlpPolicy, CnnPolicy
	"total_timesteps": int(3e5),  # 1e5 for 3x3 random/fixed, 3e5 for 5x5 fixed and 1e6 for 5x5 random
	"policy_kwargs": dict(features_extractor_class=CustomMLP,  # CustomCNN for CnnPolicy, CustomMLP for MlpPolicy
						  features_extractor_kwargs=dict(features_dim=256),
						  normalize_images=False),
	"world_size": 5,  # Supported: 3, 5
	"env_type": 'fixed',  # Supported: random, fixed, fixed_all[for debug, no learning]
	"max_steps": 25,  # 10 for 3x3, 25 for 5x5
	"penalise_steps": True,  # Set to True for -0.05 per step
	"exploration_fraction": 0.1,
	"exploration_final_eps": 0.05,
	"exploration_initial_eps": 1.0,
	"learning_starts": 50000,  # 25000 for 3x3, 50000 for 5x5
}
