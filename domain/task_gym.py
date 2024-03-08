import random
import numpy as np
import sys
from domain.make_env import make_env
from neat_src import *
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import device_get, device_put
from jax.lax import stop_gradient
from functools import partial
import gc

class GymTask():
  """Problem domain to be solved by neural network. Uses OpenAI Gym patterns.
  """ 
  def __init__(self, game, paramOnly=False, nReps=1): 
    """Initializes task environment
  
    Args:
      game - (string) - dict key of task to be solved (see domain/config.py)
  
    Optional:
      paramOnly - (bool)  - only load parameters instead of launching task?
      nReps     - (nReps) - number of trials to get average fitness
    """
    # Network properties
    self.nInput   = game.input_size
    self.nOutput  = game.output_size      
    self.actRange = game.h_act
    self.absWCap  = game.weightCap
    self.layers   = game.layers      
    self.activations = np.r_[np.full(1,1),game.i_act,game.o_act]
  
    # Environment
    self.nReps = nReps
    self.maxEpisodeLength = game.max_episode_length
    self.actSelect = game.actionSelect
    if not paramOnly:
      self.env = make_env(game.env_name)
    
    # Special needs...
    self.needsClosed = (game.env_name.startswith("CartPoleSwingUp"))    
  
  def getFitness(self, wVec, aVec, hyp=None, view=False, nRep=False, seed=-1, backprop_eval=False, gradMask=None):
    """Get fitness of a single individual.
  
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
  
    Optional:
      view    - (bool)     - view trial?
      nReps   - (nReps)    - number of trials to get average fitness
      seed    - (int)      - starting random seed for trials
  
    Returns:
      fitness - (float)    - mean reward over all trials
    """
    if nRep is False:
      nRep = self.nReps
    backprop = hyp['backprop'] if hyp and 'backprop' in hyp else False
    if not backprop:
      wVec[np.isnan(wVec)] = 0
      reward = np.empty(nRep)
      for iRep in range(nRep):
        reward[iRep] = self.testInd(wVec, aVec, view=view, seed=seed+iRep)
      fitness = np.mean(reward)
      return fitness
    else:
      wVec = np.where(np.isnan(wVec), 0, wVec)
      if not backprop_eval:
        pre_reward = 0
        for iRep in range(nRep):
          reward, wVec = self.testInd(wVec, aVec, view=view, seed=seed+iRep, hyp=hyp, backprop_eval=backprop_eval, gradMask=gradMask)
          if np.abs(reward - pre_reward) < 1e-3:
            break
          pre_reward = reward
        return reward, wVec
      else:
        reward = np.empty(nRep)
        for iRep in range(nRep):
          reward[iRep] = self.testInd(wVec, aVec, view=view, seed=seed+iRep, hyp=hyp, backprop_eval=backprop_eval)
        return np.mean(reward)
        

  def testInd(self, wVec, aVec, view=False, seed=-1, hyp=None, backprop_eval=False, gradMask=None):
    """Evaluate individual on task
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
  
    Optional:
      view    - (bool)     - view trial?
      seed    - (int)      - starting random seed for trials
  
    Returns:
      fitness - (float)    - reward earned in trial
    """
    backprop = hyp['backprop'] if hyp and 'backprop' in hyp else False
    if not backprop:
      if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
      state = self.env.reset()
      self.env.t = 0
      annOut = act(wVec, aVec, self.nInput, self.nOutput, state)  
      action = selectAct(annOut,self.actSelect)    
    
      wVec[wVec!=0]
      predName = str(np.mean(wVec[wVec!=0]))
      state, reward, done, info = self.env.step(action)
      
      if self.maxEpisodeLength == 0:
        if view:
          if self.needsClosed:
            self.env.render(close=done)  
          else:
            self.env.render()
        return reward
      else:
        totalReward = reward
      
      for tStep in range(self.maxEpisodeLength): 
        annOut = act(wVec, aVec, self.nInput, self.nOutput, state) 
        action = selectAct(annOut,self.actSelect) 
        state, reward, done, info = self.env.step(action)
        totalReward += reward  
        if view:
          if self.needsClosed:
            self.env.render(close=done)  
          else:
            self.env.render()
        if done:
          break
      return totalReward
    else:
      if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
      
      connPenalty = hyp['connPenalty'] if 'connPenalty' in hyp else 0.03
      
      if backprop_eval:
        state = self.env.trainSet
        y = self.env.target
        annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
        action = selectAct(annOut, self.actSelect)
        pred = np.where(action > 0.5, 1, 0)
        assert pred.shape == y.shape, "Prediction and target shape mismatch"
        eps = 1e-7
        action_clipped = np.clip(action, eps, 1 - eps)
        error = -np.mean(y * np.log(action_clipped) + (1 - y) * np.log(1 - action_clipped))
        # error = np.mean(np.abs(pred - y))
        nConn = np.count_nonzero(wVec)
        totalReward = -error * (1+connPenalty * np.sqrt(nConn))
        return totalReward
      
      else:
        self.env.batch = hyp['batch_size'] if 'batch_size' in hyp else 10
        state = self.env.reset()
        self.env.t = 0
        
        if jnp.ndim(wVec) < 2:
          nNodes = int(jnp.sqrt(jnp.shape(wVec)[0]))
        else:
          nNodes = int(jnp.shape(wVec)[0])
        
        def forward(wVec, aVec, input, output, state, y, actSelect, backprop, nNodes, gradMask):
            annOut = act(wVec, aVec, input, output, state, backprop, nNodes, gradMask)
            action = selectAct(annOut, actSelect, backprop)
            action = action
            assert action.shape == y.shape, "Prediction and target shape mismatch"
            eps = 1e-7 # bigger to avoid NaN
            action_clipped = jnp.clip(action, eps, 1 - eps)
            loss = -jnp.mean(y * jnp.log(action_clipped) + (1 - y) * jnp.log(1 - action_clipped))
            return loss
            
        loss = partial(forward, aVec=aVec, input=self.nInput, output=self.nOutput, actSelect=self.actSelect, backprop=backprop, nNodes=nNodes, gradMask=gradMask)
        loss = jit(loss)
        
        totalReward = 0
        done = False
        step_size = hyp['step_size'] if 'step_size' in hyp else 0.01
        grad_clip = hyp['ann_absWCap'] / 10.0
        weight_decay = hyp['weight_decay'] if 'weight_decay' in hyp else 0.001
        avg_vel = 0
        alpha = 0.99
        eps = 1e-8
        while not done:
          y = self.env.get_labels()
          grads = grad(loss)(wVec, state=state, y=y)
          # grads = jnp.where(jnp.isnan(grads), 0, grads)
          grads = jnp.clip(grads, -grad_clip, grad_clip)
          avg_vel = alpha * avg_vel + (1 - alpha) * jnp.square(grads)
          wVec = wVec - step_size * (grads / (jnp.sqrt(jnp.square(avg_vel)) + eps)) - weight_decay * wVec
          wVec = jnp.clip(wVec, -hyp['ann_absWCap'], hyp['ann_absWCap'])
          state, _, done, _ = self.env.step(None)
          # if view:
          #   if self.needsClosed:
          #     self.env.render(close=done)  
          #   else:
          #     self.env.render()
          if done:
            state = self.env.trainSet
            y = self.env.target
            wVec_np = device_get(wVec).copy()
            nConn = np.count_nonzero(wVec_np)
            annOut = act(wVec_np, aVec, self.nInput, self.nOutput, state)
            action = selectAct(annOut, self.actSelect)
            pred = np.where(action > 0.5, 1, 0)
            assert pred.shape == y.shape, "Prediction and target shape mismatch"
            # error = np.mean(np.abs(pred - y))
            eps = 1e-7
            action_clipped = np.clip(action, eps, 1 - eps)
            error = -np.mean(y * np.log(action_clipped) + (1 - y) * np.log(1 - action_clipped))
            totalReward = -error * (1+connPenalty * np.sqrt(nConn))
            break
        jax.clear_caches()
        return totalReward, wVec_np
