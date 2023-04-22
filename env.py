# Let's define a grid world
import numpy as np
import networkx as nx
from typing import List, Tuple
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os
import cv2


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
		
		# Define Actions
		self.action_space = {0: "Move Up", 1: "Move Down", 2: "Move Left", 3: "Move Right"}
		
		# Define Rewards
		self.r_goal = 1  # For reaching the goal with the marker
		self.r_obstacle = -1  # For hitting an obstacle
		self.r_step = 0.0  # For taking a step
		self.r_invalid = -1  # For reaching the goal without the marker
		# No penalty for hitting the wall
		
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
		
		# Get the positions of the entities
		robot_pos = self.get_robot_pos(state)
		marker_pos = self.get_marker_pos(state)
		obstacles_pos = self.get_obstacles_pos(state)
		goal_pos = self.get_goal_pos(state)
		
		# Create the image
		image = np.zeros((self.world_size[0], self.world_size[1], 3))
		image[robot_pos[0], robot_pos[1], :] = [1, 0, 0]  # Red
		image[marker_pos[0], marker_pos[1], :] = [0, 1, 0]  # Green
		for i in range(obstacles_pos.shape[0]):
			image[obstacles_pos[i, 0], obstacles_pos[i, 1], :] = [0, 0, 1]  # Blue
		image[goal_pos[0], goal_pos[1], :] = [1, 1, 0]  # Yellow
		
		return image
	
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
		has_marker = np.all(robot_pos == marker_pos)
		
		# First update the robot position based on the action (grid world index starts from 0 in the top left corner)
		if action == 0:
			# Move Up [If you are at the top row, then you can't move up]
			if robot_pos[0] > 0:
				robot_pos[0] -= 1
		elif action == 1:
			# Move Down [If you are at the bottom row, then you can't move down]
			if robot_pos[0] < self.world_size[0] - 1:
				robot_pos[0] += 1
		elif action == 2:
			# Move Left [If you are at the leftmost column, then you can't move left]
			if robot_pos[1] > 0:
				robot_pos[1] -= 1
		elif action == 3:
			# Move Right [If you are at the rightmost column, then you can't move right]
			if robot_pos[1] < self.world_size[1] - 1:
				robot_pos[1] += 1
		
		# Check if the robot is at the obstacle position
		at_obstacle = np.any([np.all(robot_pos == pos) for pos in obstacles_pos])
		
		# Check if the robot is at the goal position
		at_goal = np.all(robot_pos == goal_pos)
		
		if at_obstacle:
			# Robot stays in the same position -> No change in the state
			next_state = curr_state
			reward = self.r_obstacle
			done = False
			
		elif at_goal:
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
			
		return next_state, reward, done
	
	@staticmethod
	def move_robot(state, robot_pos, has_marker):
		"""
		Move the robot to the specified position
		:param state: The current state of the world
		:param robot_pos: The position to move the robot to
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
		

class ExpertPolicy:
	def __init__(self, world):
		self.world: KarelWorld = world
		self.G: nx.Graph = None
	
	def initialise(self, init_state):
		"""
		Predetermine:-
		1. Shortest path from robot to marker
		2. Shortest path from marker to goal
		:param init_state:
		:return:
		"""
		# First realise the world as a graph with each cell as a node connected to its neighbours except the obstacles
		G = nx.grid_2d_graph(init_state.shape[0], init_state.shape[1])
		obstacles_pos: np.ndarray = self.world.get_obstacles_pos(init_state)
		obstacles_pos: List[Tuple[int, int]] = [(pos[0], pos[1]) for pos in obstacles_pos]
		# Remove the obstacle node from the graph. Adjacent edges will be automatically removed
		G.remove_nodes_from(obstacles_pos)
		self.G = G
		
		
	@staticmethod
	def get_action_from_path(curr_pos, next_pos) -> int:
		"""
		:param curr_pos: The current position of the robot
		:param next_pos: The next position of the robot
		:return: The action to be taken
		"""
		if curr_pos[0] == next_pos[0]:
			if curr_pos[1] < next_pos[1]:
				return 3
			else:
				return 2
		else:
			if curr_pos[0] < next_pos[0]:
				return 1
			else:
				return 0
			
	def get_shortest_path(self, curr_pos: Tuple[int, int], next_pos: Tuple[int, int]) -> List[int]:
		"""
		:param curr_pos: The current position of the robot
		:param next_pos: The next position of the robot
		:return: The shortest path from the current position to the next position
		"""
		shortest_path = nx.shortest_path(self.G, curr_pos, next_pos, method="dijkstra")
		# Check if the shortest path is empty
		if shortest_path:
			action_path: List[int] = []
			for i in range(len(shortest_path) - 1):
				action_path.append(self.get_action_from_path(shortest_path[i], shortest_path[i + 1]))
				
			return action_path
		else:
			return []
		
	def get_all_shortest_paths(self, curr_pos: Tuple[int, int], next_pos: Tuple[int, int]) -> List[List[int]]:
		"""
		:param curr_pos: The current position of the robot
		:param next_pos: The next position of the robot
		:return: All the shortest paths from the current position to the next position
		"""
		shortest_paths = nx.all_shortest_paths(self.G, curr_pos, next_pos, method="dijkstra")
		# Check if the shortest path is empty
		if shortest_paths:
			action_paths: List[List[int]] = []
			for shortest_path in shortest_paths:
				action_path: List[int] = []
				for i in range(len(shortest_path) - 1):
					action_path.append(self.get_action_from_path(shortest_path[i], shortest_path[i + 1]))
				action_paths.append(action_path)
				
			return action_paths
		else:
			return []
		
	def get_expert_actions(self, init_state):
		
		# Initialise the world as a graph
		self.initialise(init_state)
		
		# Find the shortest path from robot to marker
		robot_pos: Tuple[int, int] = tuple(self.world.get_robot_pos(init_state))
		marker_pos: Tuple[int, int] = tuple(self.world.get_marker_pos(init_state))
		goal_pos: Tuple[int, int] = tuple(self.world.get_goal_pos(init_state))
		
		# Find the shortest path from robot to marker
		robot_to_marker_path: List[int] = self.get_shortest_path(robot_pos, marker_pos)
		
		# Find the shortest path from marker to goal
		marker_to_goal_path: List[int] = self.get_shortest_path(marker_pos, goal_pos)
		
		# Combine the two paths
		robot_to_goal_path: List[int] = robot_to_marker_path + marker_to_goal_path
		
		return robot_to_goal_path
	

def create_video_from_images(img_paths, video_name, fps=1):
	"""
	:param img_paths: The paths of the images
	:param video_name: The name of the video
	:param fps: The frames per second
	:return:
	"""
	frame = cv2.imread(img_paths[0])
	height, width, layers = frame.shape
	
	# the dimensions that cv2 expects are the opposite of numpy. Thus, size should be (size[1], size[0])
	# Pass True to the last argument to get a color video
	video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), True)
	
	for image in img_paths:
		video.write(cv2.imread(image))

	# Save the video at './video_name.mp4'
	cv2.destroyAllWindows()
	video.release()
		
	
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
	plt.figure(figsize=(world.world_size[0], world.world_size[1]))
	# Show markings for each cell in the axis
	plt.xticks(np.arange(0, world.world_size[0], 1))
	plt.yticks(np.arange(0, world.world_size[1], 1))
	# Include the legends: red for agent, green for marker, blue for obstacles, yellow for goal
	plt.legend(handles=[mpatches.Patch(color='red', label='Robot'),
						mpatches.Patch(color='green', label='Marker'),
						mpatches.Patch(color='blue', label='Obstacles'),
						mpatches.Patch(color='yellow', label='Goal')], loc='upper left')
	
	# Show the grid lines based on the world size
	plt.grid(True, which='both', axis='both')
	plt.imshow(image)
	plt.savefig("world.png")
	
	# Let's get the expert actions
	expert_policy = ExpertPolicy(world)
	actions: List[int] = expert_policy.get_expert_actions(state)
	print("Actions: ", [world.action_space[i] for i in actions])
	
	# Let's visualise the actions
	states = [world.render(state)]
	for action in actions:
		state, reward, done = world.step(action)
		states.append(world.render(state))
		if done:
			break
		
	# Let's save the images
	img_dir = "./env_renders"
	img_paths: List[str] = []
	for i in range(len(states)):
		plt.figure(figsize=(world.world_size[0], world.world_size[1]))
		# Show markings for each cell in the axis
		plt.xticks(np.arange(0, world.world_size[0], 1))
		plt.yticks(np.arange(0, world.world_size[1], 1))
		# Include the legends: red for agent, green for marker, blue for obstacles, yellow for goal
		plt.legend(handles=[mpatches.Patch(color='red', label='Robot'),
							mpatches.Patch(color='green', label='Marker'),
							mpatches.Patch(color='blue', label='Obstacles'),
							mpatches.Patch(color='yellow', label='Goal')], loc='upper left')
		
		# Show the grid lines based on the world size
		plt.grid(True, which='both', axis='both')
		plt.imshow(states[i])
		img_path = os.path.join(img_dir, "world_{}.png".format(i))
		plt.savefig(img_path)
		img_paths.append(img_path)
	
	# Let's create a video from the images
	create_video_from_images(img_paths, "./env_renders/world.mp4", fps=1)
	

if __name__ == "__main__":
	debug()
