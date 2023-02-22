# Created by Ameer Eleyan
# at 2/18/2023 10:50 PM

import itertools
import os
import random

import numpy as np


def fitness(individual, data_path):
    shingle_size, band_number = individual
    total_similarity = 0
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        similarity = find_best_params_for_file(file_path, shingle_size, band_number)
        total_similarity += similarity
    return total_similarity / len(os.listdir(data_path))


def selection(population, fitnesses, tournament_size):
    parents = []
    for i in range(len(population)):
        tournament = random.sample(range(len(population)), tournament_size)
        winner = tournament[0]
        for j in tournament:
            if fitnesses[j] > fitnesses[winner]:
                winner = j
        parents.append(population[winner])
    return parents


def crossover(parent1, parent2):
    point = random.randint(0, 1)
    child1 = [parent1[0], parent2[1]] if point == 0 else [parent2[0], parent1[1]]
    child2 = [parent2[0], parent1[1]] if point == 0 else [parent1[0], parent2[1]]
    return child1, child2


def mutation(individual, p):
    if random.random() < p:
        mutated = individual.copy()
        mutated[random.randint(0, 1)] = random.randint(2, 10)
        return mutated
    else:
        return individual


def genetic_algorithm(data_path, population_size, tournament_size, mutation_prob, num_generations):
    # Initialize population
    population = [(random.randint(2, 10), random.randint(2, 10)) for i in range(population_size)]
    best_fitness = fitness(population[0], data_path)
    best_individual = population[0]
    fitnesses = [fitness(individual, data_path) for individual in population]
    for i in range(num_generations):
        # Selection
        parents = selection(population, fitnesses, tournament_size)

        # Crossover
        offspring = []
        for j in range(0, population_size - 1, 2):
            parent1, parent2 = parents[j], parents[j + 1]
            child1, child2 = crossover(parent1, parent2)
            offspring.append(mutation(child1, mutation_prob))
            offspring.append(mutation(child2, mutation_prob))

        # Elitism
        elites = [(individual, fitness(individual, data_path)) for individual in population]
        elites.sort(key = lambda x: x[1], reverse = True)
        if elites[0][1] > best_fitness:
            best_fitness = elites[0][1]
            best_individual = elites[0][0]

        # New population
        population = [elites[i][0] for i in range(2)]
        population += [offspring[i] for i in range(2, population_size)]

        # New fitnesses
        fitnesses = [fitness(individual, data_path) for individual in population]

    return best_individual


def minhash_signature(data, num_hashes):
    # Generate random hash functions
    a = np.random.randint(1, 2 ** 20, size = num_hashes)
    b = np.random.randint(1, 2 ** 20, size = num_hashes)
    prime = 4294967311

    # Compute the signature matrix
    signature = np.full((num_hashes, data.shape[1]), np.inf)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] != 0:
                hash_values = (a * j + b) % prime % data.shape[0]
                for k in range(num_hashes):
                    if hash_values[k] < signature[k, j]:
                        signature[k, j] = hash_values[k]

    return signature


def banding(signature_matrix, band_size):
    # Divide signature matrix into bands
    num_bands = signature_matrix.shape[0] // band_size
    bands = [signature_matrix[i * band_size: (i + 1) * band_size, :] for i in range(num_bands)]
    return bands


def find_best_params_for_file(file_path, shingle_size, band_number):
    # Read file
    with open(file_path) as f:
        content = f.read()

    # Tokenize
    tokens = content.split()

    # Generate shingles
    shingles = []
    for i in range(len(tokens) - shingle_size + 1):
        shingle = ' '.join(tokens[i:i + shingle_size])
        shingles.append(shingle)

    # Create set of unique shingles
    unique_shingles = set(shingles)

    # Create binary matrix
    shingles_list = list(unique_shingles)
    binary_matrix = np.zeros((len(unique_shingles), len(shingles)))
    for i, shingle in enumerate(shingles_list):
        for j, doc_shingle in enumerate(shingles):
            if shingle == doc_shingle:
                binary_matrix[i, j] = 1

    # Compute signature matrix and bands
    signature_matrix = minhash_signature(binary_matrix, 100)
    bands = banding(signature_matrix, band_number)

    # Find candidate pairs
    candidate_pairs = set()
    for i in range(len(bands)):
        band = bands[i]
        band_dict = {}
        for j in range(band.shape[1]):
            band_col = tuple(band[:, j])
            if band_col in band_dict:
                band_dict[band_col].append(j)
            else:
                band_dict[band_col] = [j]
        for key in band_dict:
            if len(band_dict[key]) > 1:
                for pair in itertools.combinations(band_dict[key], 2):
                    candidate_pairs.add(pair)

    # Compute Jaccard similarity for candidate pairs
    max_similarity = 0
    for pair in candidate_pairs:
        similarity = np.sum(binary_matrix[:, pair[0]] & binary_matrix[:, pair[1]]) / np.sum(binary_matrix[:, pair[0]]
                                                                                            | binary_matrix[:, pair[1]])
        if similarity > max_similarity:
            max_similarity = similarity

    return max_similarity



data_path = './dataset/'
population_size = 10
tournament_size = 2
mutation_prob = 0.1
num_generations = 10
# Run genetic algorithm
best_individual = genetic_algorithm(data_path, population_size, tournament_size, mutation_prob, num_generations)
# Print best individual
print(f'Best individual: shingles = {best_individual[0]}, bands = {best_individual[1]}')
