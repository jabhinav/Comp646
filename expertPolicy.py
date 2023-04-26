from typing import (
	Dict,
	List,
	Optional,
	Union,
)
from typing import Tuple

import networkx as nx
import numpy as np


class ExpertPolicy:
	def __init__(self, world):
		self.world = world
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
	
	def get_expert_trajectory(self, state):
		# Initialise the world as a graph
		self.initialise(state)
		
		# Find the shortest path from robot to marker
		robot_pos: Tuple[int, int] = tuple(self.world.get_robot_pos(state))
		marker_pos: Tuple[int, int] = tuple(self.world.get_marker_pos(state))
		goal_pos: Tuple[int, int] = tuple(self.world.get_goal_pos(state))
		
		# Find the shortest path from robot to marker
		robot_to_marker_path: List[int] = self.get_shortest_path(robot_pos, marker_pos)
		
		# Find the shortest path from marker to goal
		marker_to_goal_path: List[int] = self.get_shortest_path(marker_pos, goal_pos)
		
		# Combine the two paths (both must be non-empty)
		if robot_to_marker_path and marker_to_goal_path:
			robot_to_goal_path: List[int] = robot_to_marker_path + marker_to_goal_path
		else:
			robot_to_goal_path: List[int] = []
		
		return robot_to_goal_path
	
	def get_expert_action(self, state):
		# Initialise the world as a graph
		self.initialise(state)
		
		# Find the shortest path from robot to marker
		robot_pos: Tuple[int, int] = tuple(self.world.get_robot_pos(state))
		marker_pos: Tuple[int, int] = tuple(self.world.get_marker_pos(state))
		goal_pos: Tuple[int, int] = tuple(self.world.get_goal_pos(state))
		
		at_goal: bool = (robot_pos == goal_pos) and (marker_pos == goal_pos)
		if at_goal:
			return None
		
		# If robot has the marker, find the shortest path from robot to goal
		has_marker: bool = robot_pos == marker_pos
		if has_marker:
			robot_to_goal_path: List[int] = self.get_shortest_path(robot_pos, goal_pos)
		
		else:
			# Find the shortest path from robot to marker
			robot_to_marker_path: List[int] = self.get_shortest_path(robot_pos, marker_pos)
			
			# Find the shortest path from marker to goal
			marker_to_goal_path: List[int] = self.get_shortest_path(marker_pos, goal_pos)
			
			# Combine the two paths (both must be non-empty)
			if robot_to_marker_path and marker_to_goal_path:
				robot_to_goal_path: List[int] = robot_to_marker_path + marker_to_goal_path
			else:
				robot_to_goal_path: List[int] = []
		
		# If the path is empty, return None
		if not robot_to_goal_path:
			return None
		else:
			return robot_to_goal_path[0]
	
	def predict(
			self,
			observation: Union[np.ndarray, Dict[str, np.ndarray]],
			state: Optional[Tuple[np.ndarray, ...]] = None,
			episode_start: Optional[np.ndarray] = None,
			deterministic: bool = False) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
		"""
			:param observation: The current state of the environment
			:param state: The last state of the environment
			:param episode_start: The first state of the environment
			:param deterministic: Whether to return deterministic action or not
			:return: The action to be taken
		"""
		# Determine if the observation is a dictionary or not
		if isinstance(observation, dict):
			observation = observation["observation"]
		
		# Determine if the observation is batched or not
		if len(observation.shape) > 3:
			batch_size = observation.shape[0]
			
			# Get the first action expert trajectory for each observation
			expert_actions = []
			for i in range(batch_size):
				expert_actions.append(self.get_expert_action(observation[i]))
			
			# Convert the list to a numpy array
			expert_actions = np.array(expert_actions)
			# Reshape the array to be of shape (batch_size, 1)
			expert_actions = expert_actions.reshape((batch_size, 1))
			
			# Return the expert actions
			return expert_actions, state
		
		else:
			# Get the expert action
			expert_action = self.get_expert_action(observation)
			
			# Return the expert action
			return np.array(expert_action), state
