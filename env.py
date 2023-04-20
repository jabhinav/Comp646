# Let's define a grid world
import numpy as np
from typing import List, Tuple


class KarelWorld:
	def __init__(self, world_size: Tuple[int, int]):
		# state space
		self.world_size = world_size
		assert self.world_size[0] == self.world_size[1], "World must be square"
		assert self.world_size[0] >= 3 and self.world_size[1] >= 3, "World must be at least 3x3"
		
		# The state of the grid world are represented as a H x W x 6 tensor.
		self.num_channels = 4
		self.empty_state = np.zeros((self.world_size[0], self.world_size[1], self.num_channels))
		
		self.max_obstacle_occupancy = 0.2
		
		# Current State
		self.state = self.empty_state.copy()
		
	def random_world(self):
		"""
		This function defines a random world. Choices made are:
		1. Randomly place the robot
		2. Randomly place the marker
		3. Randomly place the obstacles
		4. Randomly place the goal
		:return:
		"""
		# 1. Randomly place the obstacles first
		max_obstacles = max(1, int(self.world_size[0] * self.world_size[1] * self.max_obstacle_occupancy))
		num_obstacles = np.random.randint(1, max_obstacles)
		obstacles_pos = []
		for i in range(num_obstacles):
			obstacles_pos.append(np.random.randint(0, self.world_size[0], size=2))
		
		# Now place all the rest of the objects so that they don't overlap and are not on top of obstacles
		# 2. Randomly place the robot
		robot_pos = np.random.randint(0, self.world_size[0], size=2)
		while np.any([np.all(robot_pos == pos) for pos in obstacles_pos]):
			robot_pos = np.random.randint(0, self.world_size[0], size=2)
		
		# 3. Randomly place the marker
		marker_pos = np.random.randint(0, self.world_size[0], size=2)
		while np.any([np.all(marker_pos == pos) for pos in obstacles_pos]) or np.all(marker_pos == robot_pos):
			marker_pos = np.random.randint(0, self.world_size[0], size=2)
		
		# 4. Randomly place the goal
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
		
		return state
	
	def reset(self):
		"""
		Reset the world to a random state
		:return:
		"""
		self.state = self.random_world()
		return self.state
	
	def get_robot_pos(self):
		"""
		:return: the position of the robot in the state
		"""
		state = self.state
		return np.argwhere(state[:, :, 0] == 1)[0]
	
	def get_marker_pos(self):
		"""
		:return: the position of the marker in the state
		"""
		state = self.state
		return np.argwhere(state[:, :, 1] == 1)[0]
	
	def get_obstacles_pos(self):
		"""
		:return: the position of the obstacles in the state
		"""
		state = self.state
		return np.argwhere(state[:, :, 2] == 1)
	
	def get_goal_pos(self):
		"""
		:return: the position of the goal in the state
		"""
		state = self.state
		return np.argwhere(state[:, :, 3] == 1)[0]
	
	def render(self):
		"""
		Show the state of the grid world.
		Use following legends for each entity:
		R: Robot
		M: Marker
		O: Obstacle
		G: Goal
		
		:return: Image of the grid world
		"""
		
		# Get the positions of the entities
		robot_pos = self.get_robot_pos()
		marker_pos = self.get_marker_pos()
		obstacles_pos = self.get_obstacles_pos()
		goal_pos = self.get_goal_pos()
		
		# Create the image
		image = np.zeros((self.world_size[0], self.world_size[1], 3))
		image[robot_pos[0], robot_pos[1], :] = [1, 0, 0]
		image[marker_pos[0], marker_pos[1], :] = [0, 1, 0]
		for i in range(obstacles_pos.shape[0]):
			image[obstacles_pos[i, 0], obstacles_pos[i, 1], :] = [0, 0, 1]
		image[goal_pos[0], goal_pos[1], :] = [1, 1, 0]
		
		return image
	
	def step(self, action):
		"""
		:param action: The action to be performed by the robot. It can be one of the following:
		0: Move Up
		1: Move Down
		2: Move Left
		3: Move Right
		4: Pick the marker
		5: Drop the marker
		:return: The next state, reward and done flag
		"""
		curr_state = self.state
		robot_pos = self.get_robot_pos()
		marker_pos = self.get_marker_pos()
		obstacles_pos = self.get_obstacles_pos()
		goal_pos = self.get_goal_pos()
		
		# Check if the action is valid
		if action not in [0, 1, 2, 3, 4, 5]:
			raise ValueError("Invalid action")
		
		# Check if the robot is holding the marker
		robot_holding_marker = np.all(robot_pos == marker_pos)
		
		# Check if the robot is at the goal
		robot_at_goal = np.all(robot_pos == goal_pos)
		
		# Check if the robot is at the marker
		robot_at_marker = np.all(robot_pos == marker_pos)
	
	
def debug():
	# Let's create a world and save the initial state as image
	world = KarelWorld((10, 10))
	state = world.reset()
	image = world.render()
	
	robot_pos = world.get_robot_pos()
	marker_pos = world.get_marker_pos()
	obstacles_pos = world.get_obstacles_pos()
	goal_pos = world.get_goal_pos()
	print("Robot Position: ", robot_pos)
	print("Marker Position: ", marker_pos)
	print("Obstacles Position: ", obstacles_pos)
	print("Goal Position: ", goal_pos)
	
	# Lets save the image
	from matplotlib import pyplot as plt
	plt.imshow(image)
	plt.savefig("initial_state.png")
	

if __name__ == "__main__":
	debug()
