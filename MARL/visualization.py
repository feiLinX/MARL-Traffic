import gym
import os
import sys
import highway_env
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class AnimationWrapper(object):
	def __init__(self, save_floder, *args, **kwargs):
		self.Images = []
		self.Obsevations = []
		fig, ax = plt.subplots()
		self.fig = fig 
		self.ax = ax 
		self.ax.axis("off") 
		self.save_floder = save_floder
		if not os.path.exists(save_floder):
			os.mkdir(save_floder)

	def add_frame(self, obs, showim=False):
		im = self.ax.imshow(obs, animated=True)
		self.Images.append([im]) 
		if showim is True:
			self.ax.imshow(obs) 

	def save_video(self, file_name, interval=200, blit=True, repeat_delay=1000):
		ani = animation.ArtistAnimation(self.fig, self.Images, interval=interval, blit=blit,repeat_delay=repeat_delay)
		# assert file_type not in ['gif', 'mp4'], "File type error."
		ani.save(os.path.join(self.save_floder, file_name)) 