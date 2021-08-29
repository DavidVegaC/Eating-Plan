import tensorflow as tf
from data import *
import pandas as pd
import numpy as np
import random as rd

class EatingPlan(object):
    def __init__(self, total_calories,
                 population_size, generations,
                 crossover_probability, mutation_probability,
                 tournament_size, elitism):
        self.total_calories = tf.constant(total_calories, dtype='float32')
        self.N = tf.constant(len(products_tabla.index))
        self.positions = tf.constant(value=tf.range(self.N))  # posiciones del cromosoma
        self.population_size = tf.constant(population_size)
        self.generations = tf.constant(generations)
        self.crossover_probability = tf.constant(crossover_probability, dtype='float32')
        self.mutation_probability = tf.constant(mutation_probability, dtype='float32')
        self.tournament_size = tf.constant(tournament_size)
        self.elitism = tf.constant(elitism)
        self.current_generation = None
        self.fitness = None

        def operate(tensor1, tensor2):
            return tf.math.reduce_sum(tf.math.multiply(tensor1, tensor2))

        def fitness_func(genome: Genome) -> float:
            total_prot = operate(prot_data, genome)
            total_fat = operate(fat_data, genome)
            total_carb = operate(carb_data, genome)

            calories = prot_cal_p_gram * total_prot + fat_cal_p_gram * total_fat + carb_cal_p_gram * total_carb
            return tf.math.abs(calories - total_calories)

        def generate_genome() -> Genome:
            return tf.constant(
                [tf.random.uniform(shape=[1], minval=min_data[i].numpy(), maxval=max_data[i].numpy())[0].numpy() for i
                 in self.positions.numpy()], dtype='float32')

        def set_fitness(sub_population):
            hash_map = {
                "fitness": 10000000,
                "genome": tf.zeros([self.N])
            }
            for genome in sub_population:
                genome_fit = self.fitness_func(genome)
                if genome_fit < hash_map["fitness"]:
                    hash_map["fitness"] = genome_fit
                    hash_map["genome"] = genome

            return (hash_map["genome"], hash_map["fitness"])

        def random_choice(population, tournament_size: int):
            pop = population.numpy()
            (length, whatevershit) = pop.shape
            length = length - 1
            return tf.constant(value=np.array([pop[rd.randint(0, length)] for _ in range(0, tournament_size)]),
                               dtype='float32')

        def selection(population) -> Genome:
            members = random_choice(population, self.tournament_size)
            (fitness_member, fitness) = set_fitness(members)
            return fitness_member

        def crossover(parent_1: Genome, parent_2: Genome):
            length, = parent_1.shape
            genes_of_child = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
            for idx in range(length):
                n = round(rd.random(), 2)
                genes_of_child = genes_of_child.write(idx, parent_1[idx].numpy() * n + parent_2[idx].numpy() * (1 - n))

            return genes_of_child.stack()

        def mutate(individual):
            length, = individual.shape
            genes_of_mutate = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
            media, sigma = 0, 0.2
            index_1, index_2 = rd.sample(tuple(self.positions.numpy()), 2)
            r1 = round(rd.gauss(mu=media, sigma=sigma), 2)
            for idx in range(length):
                if idx == index_1 or idx == index_2:
                    genes_of_mutate = genes_of_mutate.write(idx, individual[idx].numpy() + r1)
                else:
                    genes_of_mutate = genes_of_mutate.write(idx, individual[idx].numpy())

            return tf.math.abs(genes_of_mutate.stack())

        self.fitness_func = fitness_func
        self.generate_genome = generate_genome
        self.selection = selection
        self.crossover = crossover
        self.mutate = mutate

    def best_individual(self):
        return (self.fitness[0].numpy(), self.current_generation[0].numpy())

    def rank_population(self):
        ids = tf.argsort(self.fitness)
        self.fitness = tf.gather(self.fitness, ids)
        self.current_generation = tf.gather(self.current_generation, ids)

    def calculate_population_fitness(self):
        self.fitness = tf.map_fn(self.fitness_func, self.current_generation)

    def generate_population(self):
        return tf.Variable(tf.stack([self.generate_genome() for _ in range(self.population_size.numpy())]))

    def create_new_population(self):
        new_population = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        inicio = tf.Variable(0)

        if self.elitism:
            elite = tf.identity(self.current_generation[0])
            new_population = new_population.write(inicio.numpy(), elite)
            inicio.assign_add(1)

        while new_population.size().numpy() < self.population_size.numpy():
            parent_1 = tf.identity(self.selection(self.current_generation))
            parent_2 = tf.identity(self.selection(self.current_generation))

            child_1 = parent_1

            can_crossover = rd.random() < self.crossover_probability.numpy()

            if can_crossover:
                child_1 = self.crossover(parent_1, parent_2)

            new_population = new_population.write(inicio.numpy(), child_1)
            inicio.assign_add(1)

        for idx in tf.range(1, new_population.size().numpy()):
            can_mutate = rd.random() < self.mutation_probability
            if can_mutate:
                individual = self.mutate(new_population.read(idx.numpy()))
                new_population = new_population.write(idx.numpy(), individual)

        self.current_generation = new_population.stack()

    def create_initial_population(self):
        initial_population = self.generate_population()
        self.current_generation = initial_population

    def create_first_generation(self):
        self.create_initial_population()
        self.calculate_population_fitness()
        self.rank_population()

    def create_next_generation(self):
        self.create_new_population()
        self.calculate_population_fitness()
        self.rank_population()

    def run(self):
        self.create_first_generation()
        # print("Generation 0")
        # print(self.best_individual())
        genome = None
        for idx in range(self.generations):
            self.create_next_generation()
            # print(self.best_individual())
            genome = self.current_generation[0].numpy()
        return genome