import os
from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt, patches as mpatches

from domain import KarelWorld


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


def plot_world(world: KarelWorld, img_name: str = 'world.png'):
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
	plt.savefig(img_name)


def visualise_policy(world: KarelWorld, states: List[np.ndarray], img_dir: str):
	
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
		
	# Create a video from the images
	create_video_from_images(img_paths, os.path.join(img_dir, "world.mp4"))
	
	# Clear plots
	plt.clf()
	plt.close()
