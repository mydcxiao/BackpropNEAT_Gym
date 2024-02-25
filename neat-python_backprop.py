import multiprocessing
import os
import pickle
import random
import time

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import device_get, device_put
from jax.lax import stop_gradient
from functools import partial

import gym.wrappers
import matplotlib.pyplot as plt
import numpy as np
from domain import make_env
from neat.graphs import feed_forward_layers
from neat.attributes import StringAttribute, BoolAttribute

import neat
import visualize

NUM_CORES = multiprocessing.cpu_count()


class NumpyAttribute(neat.attributes.BaseAttribute):

    _config_items = {"init_mean": [np.float32, None],
                     "init_stdev": [np.float32, None],
                     "init_type": [str, 'gaussian'],
                     "replace_rate": [np.float32, None],
                     "mutate_rate": [np.float32, None],
                     "mutate_power": [np.float32, None],
                     "max_value": [np.float32, None],
                     "min_value": [np.float32, None]}
    
    def clamp(self, value, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        return np.max(np.min(value, max_value), min_value)

    def init_value(self, config):
        mean = getattr(config, self.init_mean_name)
        stdev = getattr(config, self.init_stdev_name)
        init_type = getattr(config, self.init_type_name).lower()

        if ('gauss' in init_type) or ('normal' in init_type):
            return self.clamp(np.random.randn(mean, stdev), config)

        if 'uniform' in init_type:
            min_value = np.max(getattr(config, self.min_value_name),
                            (mean - (2 * stdev)))
            max_value = np.min(getattr(config, self.max_value_name),
                            (mean + (2 * stdev)))
            return np.random.rand(min_value, max_value)

        raise RuntimeError(f"Unknown init_type {getattr(config, self.init_type_name)!r} for {self.init_type_name!s}")

    def mutate_value(self, value, config):
        # mutate_rate is usually no lower than replace_rate, and frequently higher -
        # so put first for efficiency
        mutate_rate = getattr(config, self.mutate_rate_name)

        r = np.random.rand()
        if r < mutate_rate:
            mutate_power = getattr(config, self.mutate_power_name)
            return self.clamp(value + np.random.randn(0.0, mutate_power), config)

        replace_rate = getattr(config, self.replace_rate_name)

        if r < replace_rate + mutate_rate:
            return self.init_value(config)

        return value

    def validate(self, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        if max_value < min_value:
            raise RuntimeError("Invalid min/max configuration for {self.name}")


class BackpropNodeGene(neat.genes.DefaultNodeGene):
    
    _gene_attributes = [NumpyAttribute('bias'),
                        NumpyAttribute('response'),
                        StringAttribute('activation', options=''),
                        StringAttribute('aggregation', options='')]
    
    def __init__(self, key):
        super().__init__(key)


class BackpropConnectionGene(neat.genes.DefaultConnectionGene):
    
    _gene_attributes = [NumpyAttribute('weight'),
                        BoolAttribute('enabled')]
    
    def __init__(self, key):
        super().__init__(key)


class BackpropGenome(neat.DefaultGenome):
    
    def __init__(self, key):
        super().__init__(key)
        
        self.aggregation_function_defs = {
            'sum': lambda x: jnp.sum(x, axis=1),
            'product': lambda x: jnp.prod(x, axis=1),
            'max': lambda x: jnp.max(x, axis=1),
            'min': lambda x: jnp.min(x, axis=1),
            'mean': lambda x: jnp.mean(x, axis=1),
            'median': lambda x: jnp.median(x, axis=1),
        }
        
        self.activation_defs = {
            'sigmoid': jax.nn.sigmoid,
            'tanh': jax.nn.tanh,
            'sin': jnp.sin,
            'cos': jnp.cos,
            'gauss': lambda x: jnp.exp(-jnp.multiply(x, x) / 2.0),
            'relu': jax.nn.relu,
            'elu': jax.nn.elu,
            'lelu': jax.nn.leaky_relu,
            'selu': jax.nn.selu,
            'softplus': jax.nn.softplus,
            'identity': lambda x: x,
            'clamped': lambda x: jnp.clip(x, -1, 1),
            'inv': lambda x: 1 / x if x != 0 else x,
            'log': lambda x: jnp.log(jnp.maximum(x, 1e-7)),
            'exp': lambda x: jnp.exp(jnp.clip(x, -60, 60)),
            'abs': jnp.abs,
            'hat': lambda x: jnp.maximum(0, 1 - jnp.abs(x)),
            'square': lambda x: jnp.square(x),
            'cube': lambda x: jnp.power(x, 3),
        }
    
    # @classmethod
    # def parse_config(cls, param_dict):
    #     param_dict['node_gene_type'] = BackpropNodeGene
    #     param_dict['connection_gene_type'] = BackpropConnectionGene
    #     return neat.genome.DefaultGenomeConfig(param_dict)


class GymClassificationTask():
    
    def __init__(self, 
                 game: str = 'BackpropXOR', # 'BackpropSprial', 'BackpropGaussian', 'BackpropCircle' 
                 num_trails: int = 1, # epochs
                 num_workers: int = 1, # number of workers
                ):
        self.env = make_env(game)
        self.num_trials = num_trails
        self.num_workers = num_workers
    
    def train_genome(self, genome, config, batch=10, lr=0.01):
        self._set_batch(batch)
        prev_fitness = 0
        for _ in range(self.num_trials):
            fitness = self._train_one_epoch(genome=genome, config=config, lr=lr)
            if jnp.abs(fitness - prev_fitness) < 1e-3:
                break
        
        return float(fitness)
        # return np.float32(fitness)
    
    def evaluate_genomes(self, genomes, config, batch=10, lr=0.01):
        t0 = time.time()
        print("Training {0} epoches".format(self.num_trials))
        for _, genome in genomes:
            reward = self.train_genome(genome, config, batch, lr)
            genome.fitness = reward
        # if self.num_workers < 2:
        #     for genome in genomes:
        #         reward = self.train_genome(genome, config, batch, lr)
        #         genome.fitness = reward
        # else:
        #     with multiprocessing.Pool(self.num_workers) as pool:
        #         jobs = []
        #         for genome in genomes:
        #             jobs.append(pool.apply_async(self.train_genome,
        #                                          (genome, config, batch, lr)))

        #         for job, genome in zip(jobs, genomes):
        #             reward = job.get(timeout=None)
        #             genome.fitness = reward
        print("Final training time {0}\n".format(time.time() - t0))
    
    def _set_batch(self, batch):
        self.env.batch = batch
    
    def _train_one_epoch(self, genome, config, lr=0.01):
        self.env.t = 0
        state = self.env.reset()
        targets = self.env.get_labels()
        
        adj_list, weights, biases, responses = FeedForwardNetwork.create(genome, config)
        params = {'weights': weights, 'biases': biases, 'responses': responses}
        
        criterion = partial(loss_fn, adj_list=adj_list, genome=genome, config=config)
        
        done = False
        prev_loss = 0
        avg_vel = {k: {k1: 0 for k1 in v} for k, v in params.items()}
        alpha = 0.99
        eps = 1e-8
        while not done:
            curr_loss, grads = value_and_grad(criterion)(params, state, targets)
            avg_vel = sum(prod(avg_vel, alpha), prod(sqr(grads), 1 - alpha))
            params = sum(params, prod(div(grads, sum(sqrt(avg_vel), eps)), -lr))
            state, _, done, _ = self.env.step(None)
            targets = self.env.get_labels()
            if jnp.abs(curr_loss - prev_loss) < 1e-3:
                done = True
            if done:
                state = self.env.trainSet
                targets = self.env.target
                outputs = FeedForwardNetwork.forward(params['weights'], params['biases'], params['responses'], state, adj_list, genome, config)
                logits = jax.nn.sigmoid(outputs).reshape(-1, 1)
                pred = jnp.where(logits > 0.5, 1, 0)
                acc = jnp.mean(jnp.equal(pred, targets))
                fitness = -(acc+1e-7) * jnp.sqrt(1 + 0.03 * len(params['weights']))
                # params = jnp2np(params)
                params = jnp2float(params)
                FeedForwardNetwork.backward(params['weights'], params['biases'], params['responses'], genome)
                break
        return fitness


def prod(dic1, dic2):
    return {k: {k1: dic1[k][k1] * dic2[k][k1] for k1 in dic1[k]} for k in dic1} \
        if isinstance(dic2, dict) else {k: {k1: dic1[k][k1] * dic2 for k1 in dic1[k]} for k in dic1}


def sum(dic1, dic2):
    return {k: {k1: dic1[k][k1] + dic2[k][k1] for k1 in dic1[k]} for k in dic1} \
        if isinstance(dic2, dict) else {k: {k1: dic1[k][k1] + dic2 for k1 in dic1[k]} for k in dic1}


def div(dic1, dic2):
    return {k: {k1: dic1[k][k1] / dic2[k][k1] for k1 in dic1[k]} for k in dic1}


def sub(dic1, dic2):
    return {k: {k1: dic1[k][k1] - dic2[k][k1] for k1 in dic1[k]} for k in dic1}


def sqr(dic):
    return {k: {k1: jnp.square(dic[k][k1]) for k1 in dic[k]} for k in dic}


def sqrt(dic):
    return {k: {k1: jnp.sqrt(dic[k][k1]) for k1 in dic[k]} for k in dic}


def jnp2np(dic):
    return {k: {k1: np.float32(dic[k][k1]) for k1 in dic[k]} for k in dic}


def jnp2float(dic):
    return {k: {k1: float(dic[k][k1]) for k1 in dic[k]} for k in dic}


def loss_fn(params, inputs, targets, adj_list, genome, config):
    w = params['weights']
    b = params['biases']
    r = params['responses']
    outputs = FeedForwardNetwork.forward(w, b, r, inputs, adj_list, genome, config)
    logit = jax.nn.sigmoid(outputs).reshape(-1, 1)
    logit = jnp.clip(logit, 1e-7, 1 - 1e-7)
    loss = -jnp.mean(targets * jnp.log(logit) + (1 - targets) * jnp.log(1 - logit))
    return loss


class FeedForwardNetwork(object):
    
    def __init__(self):
        super.__init__()

    @staticmethod
    def create(genome, config):
        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]
        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        
        adj_list = []
        weights = {}
        biases = {}
        responses = {}
        for layer in layers:
            for node in layer:
                links = []
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        links.append((inode, onode))
                        weights[(inode, onode)] = genome.connections[conn_key].weight
                        biases[onode] = genome.nodes[onode].bias
                        responses[onode] = genome.nodes[onode].response
                adj_list.append((node, links))
        
        return adj_list, weights, biases, responses
    
    @staticmethod
    def backward(weights, biases, responses, genome):
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]
        nodes = [ng.key for ng in genome.nodes.values()]
        
        for conn_key in connections:
            genome.connections[conn_key].weight = weights[conn_key]
            
        for node_key in nodes:
            genome.nodes[node_key].bias = biases[node_key]
            genome.nodes[node_key].response = responses[node_key]

    @staticmethod
    def forward(weights, biases, responses, inputs, adj_list, genome, config):
        input_nodes = config.genome_config.input_keys
        output_nodes = config.genome_config.output_keys
        assert len(input_nodes) == inputs.shape[1], \
            f"Incorrect number of input nodes (Expected {len(input_nodes)}, got {inputs.shape[1]})."

        values = {key: jnp.zeros(inputs.shape[0]) for key in input_nodes + output_nodes}
        for i in range(inputs.shape[1]):
            values[input_nodes[i]] = inputs[:, i]
        
        for node, links in adj_list:
            node_inputs = jnp.zeros((inputs.shape[0], len(links)))
            for idx, (i, o) in enumerate(links):
                ng = genome.nodes[o]
                try:
                    agg_func = genome.aggregation_function_defs.get(ng.aggregation)
                except:
                    raise Exception(f"Invalid aggregation function: {ng.aggregation}")
                try:
                    act_func = genome.activation_defs.get(ng.activation)
                except:
                    raise Exception(f"Invalid activation function: {ng.activation}")
                bias = biases[o]
                response = responses[o]
                w = weights[(i, o)]
                node_inputs.at[:, idx].set(values[i] * w)
            s = agg_func(node_inputs)
            values[node] = act_func(bias + response * s)
            
            outputs = jnp.empty((inputs.shape[0], len(output_nodes)))
            for i, output in enumerate(output_nodes):
                outputs.at[:, i].set(values[output])

        return outputs


def run(args):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_backprop')
    config = neat.Config(BackpropGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(25, None))

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    ec = GymClassificationTask('BackpropXOR', 30, args.num_worker)
    while 1:
        try:
            gen_best = pop.run(ec.evaluate_genomes, 5)

            # print(gen_best)

            visualize.plot_stats(stats, ylog=False, view=False, filename="fitness.svg")

            mfs = sum(stats.get_fitness_mean()[-5:]) / 5.0
            print("Average mean fitness over last 5 generations: {0}".format(mfs))

            mfs = sum(stats.get_fitness_stat(min)[-5:]) / 5.0
            print("Average min fitness over last 5 generations: {0}".format(mfs))

            # Use the best genomes seen so far as an ensemble-ish control system.
            best_genome = stats.best_unique_genomes(1)[0]
            adj_list, weights, biases, responses = FeedForwardNetwork.create(best_genome, config)

            solved = True
            best_scores = []
            for k in range(5):
                ec.env._generate_data(type=ec.env.type)
                observation = ec.env.trainSet
                targets = ec.env.target
                score = 0
                done = False
                while not done:
                    best_action = FeedForwardNetwork.forward(weights, biases, responses, observation, adj_list, best_genome, config)
                    logits = jax.nn.sigmoid(best_action).reshape(-1, 1)
                    pred = jnp.where(logits > 0.5, 1, 0)
                    score = float(jnp.mean(jnp.equal(pred, targets)))
                    _, _, done, _ = ec.env.step(best_action)
                    # env.render()
                    if done:
                        break

                best_scores.append(score)
                avg_score = sum(best_scores) / len(best_scores)
                if avg_score > 0.1:
                    solved = False
                    break
                
            if solved:
                print("Solved.")

                # Save the winners.
                os.mkdir('./backprop_results', exist_ok=True)
                name = 'winner'
                with open(name + '.pickle', 'wb') as f:
                    pickle.dump(best_genome, f)

                visualize.draw_net(config, best_genome, view=False, filename=name + "-net.gv")
                visualize.draw_net(config, best_genome, view=False, filename=name + "-net-pruned.gv", prune_unused=True)

                break
        except KeyboardInterrupt:
            print("User break.")
            break

    # ec.env.close()


if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description=('Evolve NEAT networks'))

    parser.add_argument('-n', '--num_worker', type=int,\
    help='number of cores to use', default=8)

    args = parser.parse_args()
    
    run(args)