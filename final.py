import numpy as np
import os
import random
import gym
from Tiles.tiles import *
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys

class MountainCar:
    
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.initialise()

    def run(self):

        episodes = np.int(sys.argv[1])
        to_show = np.int(sys.argv[2])
        
        to_plot = np.zeros(len(sys.argv) - 3)
        for i in range(len(sys.argv) - 3):
            to_plot[i] = sys.argv[i + 3]

        for i in range(episodes):
            self.episodes = i
            print "Starting episode {0}".format(i)
            self.episode(False)

            if i in to_plot:
                self.plotSurface()

        programPause = raw_input("Press the <ENTER> key to continue")

        for i in range(to_show):
            self.episode(True, True)     

        if len(to_plot) > 0:
            plt.show(block=True)

    def initialise(self):
        #self.visited = []
        self.eps = 0.5 # starting exploration
        self.eps_decay = 0.9247420 # decay rate of exploration (fades to 1% exploration after 50 eps)
        self.gamma = 1
        self.lmb = 0.9
        self.tiles = 8
        self.memsize = 8192 # also number of features
        self.Theta = np.zeros(self.memsize)
        self.alpha = 0.5 / self.tiles
        self.episodes = 0
       
    def episode(self, render=True, only_exploit=False):
        self.t = 0 # reset t value
        self.e = np.zeros(self.memsize)
        self.Q_old = 0

        self.state = self.env.reset()
        self.action = self.chooseAction(copy.deepcopy(self.state))
        self.phi_activated = self.activatedTiles(copy.deepcopy(self.state), self.action)
        self.phi = np.zeros(self.memsize)
        for i in self.phi_activated:
            self.phi[i] = 1

        if render:
            self.env.render()

        while self.timestep(render, only_exploit) == False:
            self.t = self.t + 1
        print "Episode finished after {} timesteps".format(self.t)

        # diminish exploration
        self.eps = self.eps * self.eps_decay

    def timestep(self, render, only_exploit):

        # uses true online TD(lmb) method as outlined in Sutton's book (Intro to RL)
        
        # take action and observe reward (don't update observation yet!), then choose new action based on e-greedy policy
        new_state, self.reward, self.done, self.info = self.env.step(self.action)
        new_action = self.chooseAction(copy.deepcopy(self.state), only_exploit)

        self.Q_val = sum(self.phi * self.Theta)

        # same again but for the next state
        self.phi_activated_next = self.activatedTiles(copy.deepcopy(new_state), new_action)
        self.phi_next = np.zeros(self.memsize)
        for i in self.phi_activated_next:
            self.phi_next[i] = 1
        self.Q_next = sum(self.phi_next * self.Theta)

        self.e = self.gamma * self.lmb * self.e + (1 - self.alpha * self.gamma * self.lmb * sum(self.e * self.phi)) * self.phi

        self.delta = self.reward + self.gamma * self.Q_next - self.Q_val

        self.Theta = self.Theta + self.alpha * (self.delta + self.Q_val - self.Q_old) * self.e - self.alpha * (self.Q_val - self.Q_old) * self.phi 

        self.Q_old = self.Q_next
        self.phi = self.phi_next

        # update game view
        if render:
            self.env.render()

        # set new state and action to current state and action for next timestep
        self.state = new_state
        self.action = new_action

        return self.done

    def Q(self, state, action):
        activated_features = self.activatedTiles(copy.deepcopy(state), action)
        total = 0
        for q in activated_features:
            total = total + self.Theta[q]
        return total

    def chooseAction(self, state, only_exploit=False):
        e = random.random()
        if e < self.eps and only_exploit == False: # explore
            return self.env.action_space.sample()
        else: # exploit
            q_max = float("-Inf")
            a_qmax = None
            for a in range(self.env.action_space.n):
                q = self.Q(state, a)
                if q > q_max:
                    q_max = q
                    a_qmax = a
            return a_qmax

    def activatedTiles(self, observation, action):
        warped_observation = observation
        for l in range(len(warped_observation)):
            warped_observation[l] = self.tiles * warped_observation[l] / (self.env.observation_space.high[l] - self.env.observation_space.low[l])
        return tiles(self.tiles, self.memsize, warped_observation, np.array([action]))

    def plthelper(self, x, y):
        a = self.chooseAction([x, y], True) # best action in state
        q = self.Q([x, y], a)
        return -q

    def plotSurface(self):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(self.env.observation_space.low[0], self.env.observation_space.high[0], 0.01)
        y = np.arange(self.env.observation_space.low[1], self.env.observation_space.high[1], 0.002)
        X, Y = np.meshgrid(x, y)
        zs = np.array([self.plthelper(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)

        ax.plot_surface(X, Y, Z,
            rstride=2,
            cstride=2,
            cmap=cm.RdPu,
            linewidth=1,
            antialiased=True)

        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('-q')

os.system('cls' if os.name == 'nt' else 'clear')

MountainCar().run()