import os
import sys
import time
import math
import argparse
import subprocess
import numpy as np
np.set_printoptions(precision=2, linewidth=160) 

# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# prettyNeat
from neat_src import * # NEAT
from domain import *   # Task environments

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.05'


# -- Run NEAT ------------------------------------------------------------ -- #
def master(): 
  """Main NEAT optimization script
  """
  global fileName, hyp
  data = DataGatherer(fileName, hyp)
  neat = Neat(hyp)

  for gen in range(hyp['maxGen']):        
    pop = neat.ask()            # Get newly evolved individuals from NEAT
    if 'backprop' not in hyp or not hyp['backprop']:     
      reward = batchMpiEval(pop)
      neat.tell(reward)           # Send fitness to NEAT    
    else:                       
      reward, wVec = batchMpiEval(pop, backprop=True)  # Send pop to be evaluated by workers
      neat.tell(reward, wVec)           # Send fitness to NEAT    

    data = gatherData(data,neat,gen,hyp)
    print(gen, '\t - \t', data.display())

  # Clean up and data gathering at run end
  data = gatherData(data,neat,gen,hyp,savePop=True)
  data.save()
  data.savePop(neat.pop,fileName) # Save population as 2D numpy arrays
  stopAllWorkers()

def gatherData(data,neat,gen,hyp,savePop=False):
  """Collects run data, saves it to disk, and exports pickled population

  Args:
    data       - (DataGatherer)  - collected run data
    neat       - (Neat)          - neat algorithm container
      .pop     - [Ind]           - list of individuals in population    
      .species - (Species)       - current species
    gen        - (ind)           - current generation
    hyp        - (dict)          - algorithm hyperparameters
    savePop    - (bool)          - save current population to disk?

  Return:
    data - (DataGatherer) - updated run data
  """
  data.gatherData(neat.pop, neat.species)
  if (gen%hyp['save_mod']) == 0:
    data = checkBest(data)
    data.save(gen)

  if savePop is True: # Get a sample pop to play with in notebooks    
    global fileName
    pref = 'log/' + fileName
    import pickle
    with open(pref+'_pop.obj', 'wb') as fp:
      pickle.dump(neat.pop,fp)

  return data

def checkBest(data):
  """Checks better performing individual if it performs over many trials.
  Test a new 'best' individual with many different seeds to see if it really
  outperforms the current best.

  Args:
    data - (DataGatherer) - collected run data

  Return:
    data - (DataGatherer) - collected run data with best individual updated


  * This is a bit hacky, but is only for data gathering, and not optimization
  """
  global filename, hyp
  if data.newBest is True:
    bestReps = max(hyp['bestReps'], (nWorker-1))
    rep = np.tile(data.best[-1], bestReps)
    fitVector = batchMpiEval(rep, sameSeedForEachIndividual=False, backprop='backprop' in hyp and hyp['backprop'], backprop_eval=True)
    trueFit = np.mean(fitVector)
    if trueFit > data.best[-2].fitness:  # Actually better!      
      data.best[-1].fitness = trueFit
      data.fit_top[-1]      = trueFit
      data.bestFitVec = fitVector
    else:                                # Just lucky!
      prev = hyp['save_mod']
      data.best[-prev:]    = data.best[-prev]
      data.fit_top[-prev:] = data.fit_top[-prev]
      data.newBest = False
  return data


# -- Parallelization ----------------------------------------------------- -- #
def batchMpiEval(pop, sameSeedForEachIndividual=True, backprop=False, backprop_eval=False):
  """Sends population to workers for evaluation one batch at a time.

  Args:
    pop - [Ind] - list of individuals
      .wMat - (np_array) - weight matrix of network
              [N X N] 
      .aVec - (np_array) - activation function of each node
              [N X 1]

  Return:
    reward  - (np_array) - fitness value of each individual
              [N X 1]

  Todo:
    * Asynchronous evaluation instead of batches
  """
  global nWorker, hyp
  nSlave = nWorker-1
  nJobs = len(pop)
  nBatch= math.ceil(nJobs/nSlave) # First worker is master

  # Set same seed for each individual
  if sameSeedForEachIndividual is False:
    seed = np.random.randint(1000, size=nJobs)
  else:
    seed = np.random.randint(1000)

  reward = np.empty(nJobs, dtype=np.float64)
  if backprop:
    if backprop_eval:
      flag = True
    else:
      wVecs = [None for _ in range(nJobs)]
      wVecs_dims = [None for _ in range(nJobs)]
      flag = False
  i = 0 # Index of fitness we are filling
  for iBatch in range(nBatch): # Send one batch of individuals
    for iWork in range(nSlave): # (one to each worker if there)
      if i < nJobs:
        wVec   = pop[i].wMat.flatten()
        n_wVec = np.shape(wVec)[0]
        aVec   = pop[i].aVec.flatten()
        n_aVec = np.shape(aVec)[0]
        
        if backprop and not backprop_eval:
          wVecs_dims[i] = n_wVec
          gradMask = pop[i].gradMask.flatten()
        
        comm.send(n_wVec, dest=(iWork)+1, tag=1)
        comm.Send(  wVec, dest=(iWork)+1, tag=2)
        comm.send(n_aVec, dest=(iWork)+1, tag=3)
        comm.Send(  aVec, dest=(iWork)+1, tag=4)
        if sameSeedForEachIndividual is False:
          comm.send(seed.item(i), dest=(iWork)+1, tag=5)
        else:
          comm.send(  seed, dest=(iWork)+1, tag=5)
        if backprop:
          comm.send(flag, dest=(iWork)+1, tag=6)
        if backprop and not backprop_eval:
          comm.Send(gradMask, dest=(iWork)+1, tag=7) 
        
      else: # message size of 0 is signal to shutdown workers
        n_wVec = 0
        comm.send(n_wVec,  dest=(iWork)+1)
      i = i+1 
  
    # Get fitness values back for that batch
    i -= nSlave
    for iWork in range(1,nSlave+1):
      if i < nJobs:
        workResult = np.empty(1, dtype='d')
        if backprop and not backprop_eval:
          wVec = np.empty(wVecs_dims[i], dtype='d')
          comm.Recv(workResult, source=iWork, tag=1)
          reward[i] = workResult[0]
          comm.Recv(wVec, source=iWork, tag=2)
          wVecs[i] = wVec
        else:
          comm.Recv(workResult, source=iWork)
          reward[i] = workResult[0]
      i+=1
  return reward if not backprop or (backprop and backprop_eval) else (reward, wVecs)

def slave():
  """Evaluation process: evaluates networks sent from master process. 

  PseudoArgs (recieved from master):
    wVec   - (np_array) - weight matrix as a flattened vector
             [1 X N**2]
    n_wVec - (int)      - length of weight vector (N**2)
    aVec   - (np_array) - activation function of each node 
             [1 X N]    - stored as ints, see applyAct in ann.py
    n_aVec - (int)      - length of activation vector (N)
    seed   - (int)      - random seed (for consistency across workers)

  PseudoReturn (sent to master):
    result - (float)    - fitness value of network
  """
  global hyp
  task = GymTask(games[hyp['task']], nReps=hyp['alg_nReps'])

  # Evaluate any weight vectors sent this way
  while True:
    n_wVec = comm.recv(source=0,  tag=1)# how long is the array that's coming?
    if n_wVec > 0:
      wVec = np.empty(n_wVec, dtype='d')# allocate space to receive weights
      comm.Recv(wVec, source=0,  tag=2) # recieve weights

      n_aVec = comm.recv(source=0,tag=3)# how long is the array that's coming?
      aVec = np.empty(n_aVec, dtype='d')# allocate space to receive activation
      comm.Recv(aVec, source=0,  tag=4) # recieve it
      seed = comm.recv(source=0, tag=5) # random seed as int
    
      if 'backprop' not in hyp or not hyp['backprop']:
        result = task.getFitness(wVec, aVec) # process it
        comm.Send(result, dest=0)            # send it back
      else:
        backprop_eval = comm.recv(source=0, tag=6)
        if not backprop_eval:
          # Send matches Recv, send matches recv
          gradMask = np.empty(n_wVec, dtype='d')
          comm.Recv(gradMask, source=0, tag=7)
          # gradMask = comm.recv(source=0, tag=7)
          result, wVec = task.getFitness(wVec, aVec, backprop=True, gradMask=gradMask)
          comm.Send(result, dest=0, tag=1)      # send fitness back
          comm.Send(wVec, dest=0, tag=2)        # send weight vector back
        else:
          result = task.getFitness(wVec, aVec, backprop=True, backprop_eval=True)
          comm.Send(result, dest=0)      # send fitness back

    if n_wVec < 0: # End signal recieved
      print('Worker # ', rank, ' shutting down.')
      break

def stopAllWorkers():
  """Sends signal to all workers to shutdown.
  """
  global nWorker
  nSlave = nWorker-1
  print('stopping workers')
  for iWork in range(nSlave):
    comm.send(-1, dest=(iWork)+1, tag=1)

def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"
  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
    subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    return "parent"
  else:
    global nWorker, rank
    nWorker = comm.Get_size()
    rank = comm.Get_rank()
    #print('assigning the rank and nworkers', nWorker, rank)
    return "child"


# -- Input Parsing ------------------------------------------------------- -- #

def main(argv):
  """Handles command line input, launches optimization or evaluation script
  depending on MPI rank.
  """
  global fileName, hyp # Used by both master and slave processes
  fileName    = args.outPrefix
  hyp_default = args.default
  hyp_adjust  = args.hyperparam

  hyp = loadHyp(pFileName=hyp_default)
  updateHyp(hyp,hyp_adjust)

  # Launch main thread and workers
  if (rank == 0):
    master()
  else:
    slave()

if __name__ == "__main__":
  ''' Parse input and launch '''
  parser = argparse.ArgumentParser(description=('Evolve NEAT networks'))
  
  parser.add_argument('-d', '--default', type=str,\
   help='default hyperparameter file', default='p/default_neat.json')

  parser.add_argument('-p', '--hyperparam', type=str,\
   help='hyperparameter file', default=None)

  parser.add_argument('-o', '--outPrefix', type=str,\
   help='file name for result output', default='test')
  
  parser.add_argument('-n', '--num_worker', type=int,\
   help='number of cores to use', default=8)

  args = parser.parse_args()


  # Use MPI if parallel
  if "parent" == mpi_fork(args.num_worker+1): os._exit(0)

  main(args)                              
  




