import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import sys
import cv2
import math
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from PIL import Image
import io 

class BackpropClassifyEnv(gym.Env):
  """Classification as an unsupervised OpenAI Gym RL problem.
  """

  def __init__(self, trainSet, target):
    """
    Data set is a tuple of 
    [0] input data: [nSamples x nInputs]
    [1] labels:     [nSamples x 1]

    Example data sets are given at the end of this file
    """

    self.t = 0          # Current batch number
    self.t_limit = 0    # Number of batches if you need them
    self.batch   = 200 # Number of images per batch
    self.seed()
    self.viewer = None

    self.trainSet = trainSet
    self.target   = target

    nInputs = np.shape(trainSet)[1]
    high = np.array([1.0]*nInputs)
    self.action_space = spaces.Box(np.array(0,dtype=np.float32), \
                                   np.array(1,dtype=np.float32))
    self.observation_space = spaces.Box(np.array(0,dtype=np.float32), \
                                   np.array(1,dtype=np.float32))

    self.state = None
    self.trainOrder = None
    self.currIndx = None

  def seed(self, seed=None):
    ''' Randomly select from training set'''
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  
  def reset(self):
    ''' Initialize State'''    
    #print('Lucky number', np.random.randint(10)) # same randomness?
    self.trainOrder = np.random.permutation(len(self.target))
    self.t = 0 # timestep
    self.t_limit = self.target.shape[0]//self.batch - 1
    self.currIndx = self.trainOrder[self.t:self.t+self.batch]
    self.state = self.trainSet[self.currIndx,:]
    return self.state
  
  def step(self, action):
    ''' 
    Judge Classification, increment to next batch
    action - [batch x output] - softmax output
    '''
    # y = self.target[self.currIndx]

    # eps = 1e-10
    # loss = -np.mean(y * np.log(action + eps) + (1 - y) * np.log(1 - action + eps))
    # reward = -loss
    
    reward = action

    if self.t_limit > 0: # We are doing batches
      reward *= (1/self.t_limit) # average
      self.t += 1
      done = False
      if self.t >= self.t_limit:
        done = True
      self.currIndx = self.trainOrder[(self.t*self.batch):\
                                      (self.t*self.batch + self.batch)]

      self.state = self.trainSet[self.currIndx,:]
    else:
      done = True

    obs = self.state
    return obs, reward, done, {}

  def get_labels(self):
    return self.target[self.currIndx]
  
  def render(self, weight, mode='rgb_array', close=False):
    #TODO - render the diagram and save it to gif
    return None


# -- Data Sets ----------------------------------------------------------- -- #

def XOR(num=None, noise=None):
    ''' 
    XOR data set
    '''
    if num == None: num = 200
    if noise == None: noise = 0.5
    x = np.random.uniform(-5,5,size=(num, 2))
    x = x + np.random.normal(0,noise,(num,2))
    y = np.zeros(num)
    y[np.where((x[:,0]>0)&(x[:,1]>0))[0]] = 1
    y[np.where((x[:,0]<0)&(x[:,1]<0))[0]] = 1
    return x, y.reshape(-1,1)

def spiral(num=None, noise=None):
    ''' 
    Spiral data set
    '''
    if num == None: num = 200
    if noise == None: noise = 0.5
    r = np.linspace(0,1,num//2, endpoint=False) * 6.0
    tp = np.linspace(0,1.75*2*np.pi,num//2, endpoint=False)
    tn = np.linspace(0,1.75*2*np.pi,num//2, endpoint=False) + np.pi
    xp = np.array([r*np.sin(tp), r*np.cos(tp)]).T + np.random.uniform(-noise,noise,(num//2,2))
    xn = np.array([r*np.sin(tn), r*np.cos(tn)]).T + np.random.uniform(-noise,noise,(num//2,2))
    x = np.concatenate((xp,xn))
    y = np.zeros(num)
    y[num//2:] = 1
    return x, y.reshape(-1,1)
  
def gaussian(num=None, noise=None):
    ''' 
    Gaussian data set
    '''
    if num == None: num = 200
    if noise == None: noise = 0.5
    xp = np.random.normal((2,2),noise+1,(num//2,2))
    xn = np.random.normal((-2,-2),noise+1,(num//2,2))
    x = np.concatenate((xp,xn))
    y = np.zeros(num)
    y[:num//2] = 1
    return x, y.reshape(-1,1)

def circle(num=None, noise=None):
    ''' 
    Circle data set
    '''
    if num == None: num = 200
    if noise == None: noise = 0.5
    radius = 5.0
    def getLabel(x):
        if x[0]**2 + x[1]**2 < (radius*0.5)**2:
            return 1
        else:
            return 0
    rp = np.random.uniform(0,radius*0.5,num//2)
    anglep = np.random.uniform(0,2*np.pi,num//2)
    xp0 = rp * math.sin(anglep)
    xp1 = rp * math.cos(anglep)
    noise = np.random.uniform(-radius, radius, (num//2,2)) * noise / 3
    xp = np.array([xp0, xp1]).T + noise
    rn = np.random.uniform(radius*0.75,radius,num//2)
    anglen = np.random.uniform(0,2*np.pi,num//2)
    xn0 = rn * math.sin(anglen)
    xn1 = rn * math.cos(anglen)
    noise = np.random.uniform(-radius, radius, (num//2,2)) * noise / 3
    xn = np.array([xn0, xn1]).T + noise
    x = np.concatenate((xp,xn))
    y = np.zeros(num)
    y[:num//2] = 1
    return x, y.reshape(-1,1)

 
