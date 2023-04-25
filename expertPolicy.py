from typing import List, Tuple

import networkx as nx
import numpy as np

from domain import KarelWorld


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
