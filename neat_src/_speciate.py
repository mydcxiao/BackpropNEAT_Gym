import numpy as np
from utils import *

class Species():
  """Species class, only contains fields: all methods belong to the NEAT class.
  Note: All 'species' related functions are part of the Neat class, though defined in this file.
  """

  def __init__(self,seed):
    """Intialize species around a seed
    Args:
      seed - (Ind) - individual which anchors seed in compatibility space

    Attributes:
      seed       - (Ind)   - individual who acts center of species
      members    - [Ind]   - individuals in species
      bestInd    - (Ind)   - highest fitness individual ever found in species
      bestFit    - (float) - highest fitness ever found in species
      lastImp    - (int)   - generations since a new best individual was found
      nOffspring - (int)   - new individuals to create this generation
    """
    self.seed = seed      # Seed is type Ind
    self.members = [seed] # All inds in species
    self.bestInd = seed
    self.bestFit = seed.fitness
    self.lastImp = 0
    self.nOffspring = []

def speciate(self):  
  """Divides population into species and assigns each a number of offspring/
  """
  # Readbility
  p = self.p
  pop = self.pop
  species = self.species

  if p['alg_speciate'] == 'neat':
    # Adjust species threshold to track desired number of species
    if len(species) > p['spec_target']:
      p['spec_thresh'] += p['spec_compatMod']

    if len(species) < p['spec_target']:
      p['spec_thresh'] -= p['spec_compatMod']

    if p['spec_thresh'] < p['spec_threshMin']:
      p['spec_thresh'] = p['spec_threshMin']
    
    species, pop = self.assignSpecies  (species, pop, p)
    species      = self.assignOffspring(species, pop, p)

  elif p['alg_speciate'] == "none": 
    # Recombination takes a species, when there is no species we dump the whole population into one species that is awarded all offspring
    species = [Species(pop[0])]
    species[0].nOffspring = p['popSize']
    for ind in pop:
      ind.species = 0
    species[0].members = pop

  # Update
  self.p = p
  self.pop = pop
  self.species = species

def assignSpecies(self, species, pop, p):
  """Assigns each member of the population to a species.
  Fills each species class with nearests members, assigns a species Id to each
  individual for record keeping

  Args:
    species - (Species) - Previous generation's species
      .seed       - (Ind) - center of species
    pop     - [Ind]     - unassigned individuals
    p       - (Dict)    - algorithm hyperparameters

  Returns:
    species - (Species) - This generation's species
      .seed       - (Ind) - center of species
      .members    - [Ind] - parent population
    pop     - [Ind]     - individuals with species ID assigned

  """

  # Get Previous Seeds
  if len(species) == 0:
      # Create new species if none exist
      species = [Species(pop[0])]
      species[0].nOffspring = p['popSize']
      species[0].members = []
  else:
    # Remove existing members
    # for iSpec in range(len(species)):
    #   species[iSpec].members = []
    # Create new seeds
    unspeciated = set(range(len(pop)))
    for iSpec in range(len(species)):
      candidates = []
      for gid in unspeciated:
        g = pop[gid]
        d = self.compatDist(species[iSpec].seed.conn, g.conn)
        candidates.append((d, gid))
      # The new representative is the genome closest to the current representative.
      _, new_seed_id = min(candidates, key=lambda x: x[0])
      new_seed = pop[new_seed_id]
      species[iSpec].seed = new_seed
      species[iSpec].bestFit = new_seed.fitness
      species[iSpec].bestInd = new_seed
      # species[iSpec].members = [new_seed]
      species[iSpec].members = []
      unspeciated.remove(new_seed_id)

  assert p['spec_thresh'] > 0, "ERROR: Species threshold must be positive"
  # Assign members of population to first species within compat distance
  for i in range(len(pop)):
    candidates = []
    iSpec = 0
    while iSpec < len(species):
      ref = np.copy(species[iSpec].seed.conn)
      ind = np.copy(pop[i].conn)
      cDist = self.compatDist(ref,ind)
      if cDist < p['spec_thresh']:
        candidates.append((cDist, iSpec))
      iSpec += 1
    # find best species to assign to
    if len(candidates) > 0:
      _, best_iSpec = min(candidates, key=lambda x: x[0])
      pop[i].species = best_iSpec
      species[best_iSpec].members.append(pop[i])
    # If no seed is close enough, start your own species
    else:
      pop[i].species = iSpec
      species.append(Species(pop[i]))
  
  return species, pop

def assignOffspring(self, species, pop, p):
  """Assigns number of offspring to each species based on fitness sharing.
  NOTE: Ordinal rather than the cardinal fitness of canonical NEAT is used.

  Args:
    species - (Species) - this generation's species
      .members    - [Ind]   - individuals in species
    pop     - [Ind]     - individuals with species assigned
      .fitness    - (float) - performance on task (higher is better)
    p       - (Dict)    - algorithm hyperparameters

  Returns:
    species - (Species) - This generation's species
      .nOffspring - (int) - number of children to produce
  """

  nSpecies = len(species)
  if nSpecies == 1:
    species[0].nOffspring = p['popSize']
  else:
    # -- Fitness Sharing
    # Rank all individuals
    popFit = np.asarray([ind.fitness for ind in pop])
    popRank = tiedRank(popFit)
    if p['select_rankWeight'] == 'exp':
      rankScore = 1/popRank
    elif p['select_rankWeight'] == 'lin':
      rankScore = 1+abs(popRank-len(popRank))
    else:
      print("Invalid rank weighting (using linear)")
      rankScore = 1+abs(popRank-len(popRank))
    specId = np.asarray([ind.species for ind in pop])

    # Best and Average Fitness of Each Species
    speciesFit = np.zeros((nSpecies,1))
    speciesTop = np.zeros((nSpecies,1))
    for iSpec in range(nSpecies):
      if not np.any(specId==iSpec):
        speciesFit[iSpec] = 0
      else:
        speciesFit[iSpec] = np.mean(rankScore[specId==iSpec])
        speciesTop[iSpec] = np.max(popFit[specId==iSpec])

        # Did the species improve?
        if speciesTop[iSpec] > species[iSpec].bestFit:
          species[iSpec].bestFit = speciesTop[iSpec]
          bestId = np.argmax(popFit[specId==iSpec])
          species[iSpec].bestInd = species[iSpec].members[bestId]
          species[iSpec].lastImp = 0
        else:
          species[iSpec].lastImp += 1

        # Stagnant species don't recieve species fitness
        if species[iSpec].lastImp > p['spec_dropOffAge']:
          speciesFit[iSpec] = 0
          
    # -- Assign Offspring
    if sum(speciesFit) == 0:
      speciesFit = np.ones((nSpecies,1))
      print("WARN: Entire population stagnant, continuing without extinction")
      
    offspring = bestIntSplit(speciesFit, p['popSize'])
    for iSpec in range(nSpecies):
      species[iSpec].nOffspring = offspring[iSpec]
  
  # Extinction    
  species[:] = [s for s in species if s.nOffspring != 0]
  
  return species

def compatDist(self, ref, ind):
  """Calculate 'compatiblity distance' between to genomes

  Args:
    ref - (np_array) -  reference genome connection genes
          [5 X nUniqueGenes]
          [0,:] == Innovation Number (unique Id)
          [3,:] == Weight Value
    ind - (np_array) -  genome being compared
          [5 X nUniqueGenes]
          [0,:] == Innovation Number (unique Id)
          [3,:] == Weight Value

  Returns:
    dist - (float) - compatibility distance between genomes
  """

  # Find matching genes
  IA, IB = quickINTersect(ind[0,:].astype(int),ref[0,:].astype(int))          
  
  # Calculate raw genome distances
  ind[3,np.isnan(ind[3,:])] = 0
  ref[3,np.isnan(ref[3,:])] = 0
  weightDiff = abs(ind[3,IA] - ref[3,IB])
  geneDiff   = sum(np.invert(IA)) + sum(np.invert(IB))

  # Normalize and take weighted sum
  nInitial = self.p['ann_nInput'] + self.p['ann_nOutput']
  longestGenome = max(len(IA),len(IB)) - nInitial
  # weightDiff = np.mean(weightDiff)
  weightDiff = np.mean(weightDiff) / (1 + np.max(weightDiff))
  geneDiff   = geneDiff   / (1+longestGenome) # this can be bigger than 1 but less than 2
  # geneDiff = geneDiff / (len(IA) + len(IB) - 2*nInitial)

  dist = geneDiff   * self.p['spec_geneCoef']      \
       + weightDiff * self.p['spec_weightCoef']  
  return dist
