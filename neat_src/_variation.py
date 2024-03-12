import numpy as np
import itertools
from .ind import Ind
from utils import *


def evolvePop(self):
  """ Evolves new population from existing species.
  Wrapper which calls 'recombine' on every species and combines all offspring into a new population. When speciation is not used, the entire population is treated as a single species.
  """
  newPop = []
  worst = min(range(len(self.species)), key=lambda i: self.species[i].bestInd.fitness)
  best = max(range(len(self.species)), key=lambda i: self.species[i].bestInd.fitness)
  for i in range(len(self.species)):
    children, self.innov = self.recombine(self.species[i],\
                           self.innov, self.gen, i if i==worst else None, best=best)
    newPop.append(children)
  self.pop = list(itertools.chain.from_iterable(newPop))   
  
def recombine(self, species, innov, gen, worst=None, best=None):
  """ Creates next generation of child solutions from a species

  Procedure:
    ) Sort all individuals by rank
    ) Eliminate lower percentage of individuals from breeding pool
    ) Pass upper percentage of individuals to child population unchanged
    ) Select parents by tournament selection
    ) Produce new population through crossover and mutation

  Args:
      species - (Species) -
        .members    - [Ind] - parent population
        .nOffspring - (int) - number of children to produce
      innov   - (np_array)  - innovation record
                [5 X nUniqueGenes]
                [0,:] == Innovation Number
                [1,:] == Source
                [2,:] == Destination
                [3,:] == New Node?
                [4,:] == Generation evolved
      gen     - (int) - current generation

  Returns:
      children - [Ind]      - newly created population
      innov   - (np_array)  - updated innovation record

  """
  p = self.p
  nOffspring = int(species.nOffspring)
  pop = species.members
  children = []
  
  extincted = False
  if worst is not None:
    worst_fit, best_fit = self.species[worst].bestInd.fitness, self.species[best].bestInd.fitness
    worst_nConn, best_nConn = self.species[worst].bestInd.nConn, self.species[best].bestInd.nConn
    if abs(worst_fit / (1 + p['connPenalty'] * np.sqrt(worst_nConn))) > 1.2 * abs(best_fit / (1 + p['connPenalty'] * np.sqrt(best_nConn))):
      prob = p['spec_extinctProb'] if 'spec_extinctProb' in p else 0
      if np.random.rand() < prob:
        pop = self.species[best].members
        self.species[worst].seed = self.species[best].seed
        extincted = True
  
  # Sort by rank
  pop.sort(key=lambda x: x.rank)

  # Cull  - eliminate worst individuals from breeding pool
  numberToCull = int(np.floor(p['select_cullRatio'] * len(pop)))
  if numberToCull > 0:
    pop[-numberToCull:] = []     

  if not extincted:
    # Elitism - keep best individuals unchanged
    # nElites = int(np.floor(len(pop)*p['select_eliteRatio']))
    nElites = int(np.ceil(len(pop)*p['select_eliteRatio']))
    for i in range(nElites):
      children.append(pop[i])
      nOffspring -= 1
  
  if nOffspring > 0: #TODO: why is this necessary?
    # Get parent pairs via tournament selection
    # -- As individuals are sorted by fitness, index comparison is 
    # enough. In the case of ties the first individual wins    
    
    parentA = np.random.randint(len(pop),size=(nOffspring,p['select_tournSize'])) 
    parentB = np.random.randint(len(pop),size=(nOffspring,p['select_tournSize']))  
    parents = np.vstack( (np.min(parentA,1), np.min(parentB,1) ) )
    parents = np.sort(parents,axis=0) # Higher fitness parent first    
    
    # Breed child population
    for i in range(nOffspring):
      if np.random.rand() > p['prob_crossover']:
        # Mutation only: take only highest fit parent
        child, innov = pop[parents[0,i]].createChild(p,innov,gen)
      else:
        # Crossover
        child, innov = pop[parents[0,i]].createChild(p,innov,gen,\
                           mate=pop[parents[1,i]])

      child.express()
      children.append(child)      
  
  return children, innov
