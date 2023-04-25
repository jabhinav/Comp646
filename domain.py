from typing import Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class KarelWorld:
	def __init__(self, world_size: Tuple[int, int], env_type: str = "random"):
		# state space
		self.world_size = world_size
		assert self.world_size[0] == self.world_size[1], "World must be square"
		assert self.world_size[0] >= 3 and self.world_size[1] >= 3, "World must be at least 3x3"
		
		self.max_obstacle_occupancy = 0.4
		
		# # Current State
		self.num_channels = 4
		self.empty_state = np.zeros((self.world_size[0], self.world_size[1], self.num_channels))
		self.state = self.empty_state.copy()
		
		# # Define Actions
		self.action_space = {0: "Move Up", 1: "Move Down", 2: "Move Left", 3: "Move Right"}
		
		# # Define Rewards
		self.r_goal = 1  # For reaching the goal with the marker
		self.r_marker = 1  # For picking up the marker
		self.r_step = -0.05  # For taking a step
		self.r_obstacle = -1  # For hitting an obstacle or the wall
		self.r_invalid = -1  # For reaching the goal without the marker
	
		# # Environment Type
		self.env_type = env_type
		
		# Other Flags
		self.marker_picked = False
		
	def random_world(self, obstacles_pos=None, robot_pos=None, marker_pos=None, goal_pos=None):
		"""
		This function defines a random world. Choices made are:
		1. Randomly place the robot
		2. Randomly place the marker
		3. Randomly place the obstacles
		4. Randomly place the goal
		:return: state, info
		"""
		
		if obstacles_pos is None:
			# 1. Randomly place the obstacles first
			max_obstacles = max(1, int(self.world_size[0] * self.world_size[1] * self.max_obstacle_occupancy))
			num_obstacles = np.random.randint(1, max_obstacles) if max_obstacles > 1 else 1
			obstacles_pos = []
			for i in range(num_obstacles):
				obstacles_pos.append(np.random.randint(0, self.world_size[0], size=2))
		else:
			num_obstacles = len(obstacles_pos)
		
		# Now place all the rest of the objects so that they don't overlap and are not on top of obstacles
		# 2. Randomly place the robot
		if robot_pos is None:
			robot_pos = np.random.randint(0, self.world_size[0], size=2)
			while np.any([np.all(robot_pos == pos) for pos in obstacles_pos]):
				robot_pos = np.random.randint(0, self.world_size[0], size=2)
		
		# 3. Randomly place the marker
		if marker_pos is None:
			marker_pos = np.random.randint(0, self.world_size[0], size=2)
			while np.any([np.all(marker_pos == pos) for pos in obstacles_pos]) or np.all(marker_pos == robot_pos):
				marker_pos = np.random.randint(0, self.world_size[0], size=2)
		
		# 4. Randomly place the goal
		if goal_pos is None:
			goal_pos = np.random.randint(0, self.world_size[0], size=2)
			while np.any([np.all(goal_pos == pos) for pos in obstacles_pos]) or np.all(goal_pos == robot_pos) or np.all(
					goal_pos == marker_pos):
				goal_pos = np.random.randint(0, self.world_size[0], size=2)
		
		# 5. Create the state
		state = self.empty_state.copy()
		state[robot_pos[0], robot_pos[1], 0] = 1
		state[marker_pos[0], marker_pos[1], 1] = 1
		for i in range(num_obstacles):
			state[obstacles_pos[i][0], obstacles_pos[i][1], 2] = 1
		state[goal_pos[0], goal_pos[1], 3] = 1
		
		info = {
			"robot_pos": robot_pos,
			"marker_pos": marker_pos,
			"obstacles_pos": obstacles_pos,
			"goal_pos": goal_pos
		}
		
		return state, info
	
	def reset(self):
		"""
		Reset the world to a random state
		:return:
		"""
		
		# Reset Flags
		self.marker_picked = False
		
		if self.env_type == "random":
			self.state, info = self.random_world()
		
		elif self.env_type == "fixed_all":
			# Specify and fix obstacles
			if self.world_size == (7, 7):
				obstacles_pos = [np.array([0, 0]), np.array([1, 1]), np.array([5, 5])]
				robot_pos = np.array([3, 5])
				marker_pos = np.array([4, 0])
				goal_pos = np.array([5, 6])
			elif self.world_size == (5, 5):
				obstacles_pos = [np.array([1, 3]), np.array([2, 0]), np.array([3, 2]), np.array([4, 0]), np.array([4, 3])]
				robot_pos = np.array([0, 4])
				marker_pos = np.array([3, 4])
				goal_pos = np.array([1, 2])
			elif self.world_size == (3, 3):
				obstacles_pos = [np.array([1, 1])]
				robot_pos = np.array([0, 1])
				marker_pos = np.array([0, 0])
				goal_pos = np.array([2, 0])
			else:
				raise ValueError("Specify fixed obstacles for this world size : {}".format(self.world_size))
			
			self.state, info = self.random_world(obstacles_pos=obstacles_pos, robot_pos=robot_pos,
												 marker_pos=marker_pos, goal_pos=goal_pos)
		
		elif self.env_type == "fixed_obs":
			# Specify and fix obstacles
			if self.world_size == (7, 7):
				obstacles_pos = [np.array([0, 0]), np.array([1, 1]), np.array([5, 5])]
			elif self.world_size == (5, 5):
				obstacles_pos = [np.array([1, 3]), np.array([2, 0]), np.array([3, 2]), np.array([4, 0]), np.array([4, 3])]
			elif self.world_size == (3, 3):
				obstacles_pos = [np.array([1, 1])]
			else:
				raise ValueError("Specify fixed obstacles for this world size : {}".format(self.world_size))
			
			self.state, info = self.random_world(obstacles_pos=obstacles_pos)
		
		else:
			raise ValueError("Invalid environment type : {}".format(self.env_type))
		
		return self.state, info
	
	@staticmethod
	def move_robot(state, robot_pos, has_marker):
		"""
		Move the robot to the specified position
		:param state: The current state of the world
		:param robot_pos: The position to move the robot to
		:param has_marker: Whether the robot has the marker or not
		:return: The next state
		"""
		# Reset the current robot position
		state[:, :, 0] = 0
		
		# Update the robot position
		state[robot_pos[0], robot_pos[1], 0] = 1
		
		# If the robot has the marker, then update the marker position and move the marker to the robot position
		if has_marker:
			state[:, :, 1] = 0
			state[robot_pos[0], robot_pos[1], 1] = 1
			state[robot_pos[0], robot_pos[1], 2] = 0
		
		return state
	
	def step(self, action):
		"""
		How to update the state of the world based on the action performed by the robot
		- If the action is to move up, down, left or right, then the robot moves to the corresponding position
		- If the action takes you to the marker position, you automatically pick up the marker
		- If the action takes you to the goal position, you automatically drop the marker
		- If the action takes you to the goal position with the marker, then you win the game and the episode ends
		- If the action takes you to the goal position without the marker, then you lose the game and the episode ends
		- If the action takes you to the obstacle position, then robot stays in the same position and gets a negative reward

		:param action: The action to be performed by the robot. It can be one of the following:
		0: Move Up
		1: Move Down
		2: Move Left
		3: Move Right
		:return: The next state, reward and done flag
		"""
		curr_state = self.state
		robot_pos = self.get_robot_pos()
		marker_pos = self.get_marker_pos()
		obstacles_pos = self.get_obstacles_pos()
		goal_pos = self.get_goal_pos()
		
		# Check if the action is valid
		if action not in [0, 1, 2, 3]:
			raise ValueError("Invalid action")
		
		orig_robot_pos = robot_pos.copy()
		has_marker = np.all(robot_pos == marker_pos)  # This will be true if the robot reaches or has the marker
		
		hit_wall = False
		
		# # First update the robot position based on the action (grid world index starts from 0 in the top left corner)
		# Move Up [If you are at the top row, then you can't move up]
		if action == 0:
			if robot_pos[0] > 0:
				robot_pos[0] -= 1
			else:
				hit_wall = True
		# Move Down [If you are at the bottom row, then you can't move down]
		elif action == 1:
			if robot_pos[0] < self.world_size[0] - 1:
				robot_pos[0] += 1
			else:
				hit_wall = True
		# Move Left [If you are at the leftmost column, then you can't move left]
		elif action == 2:
			if robot_pos[1] > 0:
				robot_pos[1] -= 1
			else:
				hit_wall = True
		# Move Right [If you are at the rightmost column, then you can't move right]
		elif action == 3:
			if robot_pos[1] < self.world_size[1] - 1:
				robot_pos[1] += 1
			else:
				hit_wall = True
		else:
			raise ValueError("Invalid action: {}".format(action))
		
		# Check if the robot is at the obstacle position
		hit_obstacle = np.any([np.all(robot_pos == pos) for pos in obstacles_pos])
		
		# Check if the robot is at the goal position
		reached_goal = np.all(robot_pos == goal_pos)
		
		if hit_obstacle or hit_wall:
			# Robot stays in the same position -> No change in the state
			next_state = curr_state
			reward = self.r_obstacle
			done = False
		
		elif reached_goal:
			# Check if the robot has the marker
			if has_marker:
				# Robot wins the game ->
				# Since the robot has the marker and moves to the goal position, the marker is dropped
				next_state = self.move_robot(curr_state, robot_pos, has_marker)
				reward = self.r_goal
				done = True
			else:
				# Robot loses the game ->
				# Since the robot does not have the marker and moves to the goal position, the marker is not dropped
				next_state = self.move_robot(curr_state, orig_robot_pos, has_marker)
				reward = self.r_invalid
				done = True
		else:
			# Robot moves to the new position
			next_state = self.move_robot(curr_state, robot_pos, has_marker)
			reward = self.r_step
			done = False
		
		info = {
			"robot_pos": robot_pos,
			"marker_pos": marker_pos,
			"obstacles_pos": obstacles_pos,
			"goal_pos": goal_pos,
			"has_marker": has_marker,
			"hit_obstacle": hit_obstacle,
			"hit_wall": hit_wall,
			"reached_goal": reached_goal
		}
		
		return next_state, reward, done, info
	
	def render(self, state=None):
		"""
		Show the state of the grid world.
		Use following legends for each entity:
		R: Robot
		M: Marker
		O: Obstacle
		G: Goal

		:return: Image of the grid world
		"""
		if state is None:
			state = self.state
		
		# Get the positions of the entities
		robot_pos = self.get_robot_pos(state)
		marker_pos = self.get_marker_pos(state)
		obstacles_pos = self.get_obstacles_pos(state)
		goal_pos = self.get_goal_pos(state)
		
		# Create the image
		image = np.zeros((self.world_size[0], self.world_size[1], 3), dtype=np.float32)
		image[robot_pos[0], robot_pos[1], :] = [1, 0, 0]  # Red
		image[marker_pos[0], marker_pos[1], :] = [0, 1, 0]  # Green
		for i in range(obstacles_pos.shape[0]):
			image[obstacles_pos[i, 0], obstacles_pos[i, 1], :] = [0, 0, 1]  # Blue
		image[goal_pos[0], goal_pos[1], :] = [1, 1, 0]  # Yellow
		
		return image
	
	def get_robot_pos(self, state=None):
		"""
		:return: the position of the robot in the state
		"""
		if state is None:
			state = self.state
		return np.argwhere(state[:, :, 0] == 1)[0]
	
	def get_marker_pos(self, state=None):
		"""
		:return: the position of the marker in the state
		"""
		if state is None:
			state = self.state
		return np.argwhere(state[:, :, 1] == 1)[0]
	
	def get_obstacles_pos(self, state=None):
		"""
		:return: the position of the obstacles in the state
		"""
		if state is None:
			state = self.state
		return np.argwhere(state[:, :, 2] == 1)
	
	def get_goal_pos(self, state=None):
		"""
		:return: the position of the goal in the state
		"""
		if state is None:
			state = self.state
		return np.argwhere(state[:, :, 3] == 1)[0]


class GymKarelWorld(gym.Env):
	"""
	Environment for Karel in Gym. Same as the KarelWorld class but with Gym interface
	To verify the environment, run the following command:
	from stable_baselines3.common.env_checker import check_env
	check_env(GymKarelWorld((7, 7), env_type='fixed'))
	"""
	metadata = {'render.modes': ['human']}
	
	def __init__(self, world_size: tuple, env_type: str = 'fixed', max_steps: int = 100):
		"""
		:param world_size: The size of the world
		:param env_type: The type of the environment. Can be either 'fixed' or 'random'
		"""
		super(GymKarelWorld, self).__init__()
		self.world: KarelWorld = KarelWorld(world_size, env_type)
		self.action_space = spaces.Discrete(4)
		self.observation_space = spaces.Box(low=0, high=1,
											shape=(world_size[0], world_size[1], self.world.num_channels),
											dtype=np.float32)
		self.max_steps = max_steps
		self.curr_step = 0
		
	def step(self, action: np.ndarray):
		"""
		:param action: The action to be taken
		:return: The next state, reward, done, info
		"""
		truncated = False
		
		# Convert action to int if np.ndarray using argmax and shape == (action_space.n,) else just use the action using int
		if isinstance(action, np.ndarray) and action.shape == (self.action_space.n,):
			action = int(np.argmax(action))
		elif isinstance(action, np.ndarray) and action.shape == (1,):
			action = int(action[0])
		elif isinstance(action, np.ndarray) and action.shape == ():
			action = int(action)
		
		state, reward, done, info = self.world.step(action)
		
		# Check if the episode is done or truncated due to max steps (set done to True)
		self.curr_step += 1
		if self.curr_step >= self.max_steps:
			truncated = True if not done else False
			done = True
		
		return state, reward, truncated, done, info
	
	def reset(self):
		"""
		:return: The initial state
		"""
		self.curr_step = 0
		state, info = self.world.reset()
		return state, info
	
	def render(self, mode='human'):
		"""
		:param mode: The mode to render the environment
		:return: The rendered environment
		"""
		return self.world.render()
	
	def close(self):
		"""
		Close the environment
		:return:
		"""
		pass
