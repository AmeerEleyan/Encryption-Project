# Created by Ameer Eleyan
# at 2/16/2023 9:24 PM

import numpy as np
from collections import defaultdict
import random


def create_shingles(data, shingle_size):
    shingles = []
    for d in data:
        s = set()
        for i in range(len(d) - shingle_size + 1):
            shingle = tuple(d[i:i + shingle_size])
            s.add(shingle)
        shingles.append(s)
    return shingles


def create_bands(shingles, num_bands):
    bands = []
    for s in shingles:
        b = []
        for i in range(num_bands):
            band = frozenset(list(s)[i * num_bands:(i + 1) * num_bands])
            b.append(band)
        bands.append(b)
    return bands


def generate_hash_codes(bands):
    hash_codes = []
    for b in bands:
        hc = []
        for band in b:
            h = hash(band)
            hc.append(h)
        hash_codes.append(hc)
    return np.array(hash_codes)


def find_similar_pairs(data, hash_codes, threshold):
    similar_pairs = set()
    buckets = defaultdict(list)
    for i, hc in enumerate(hash_codes):
        for j, h in enumerate(hc):
            buckets[h].append(i)
    for bucket in buckets.values():
        if len(bucket) > 1:
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    d1 = data[bucket[i]]
                    d2 = data[bucket[j]]
                    if jaccard_similarity(d1, d2) >= threshold:
                        similar_pairs.add((bucket[i], bucket[j]))
    return similar_pairs


def jaccard_similarity(s1, s2):
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    return len(s1.intersection(s2)) / len(s1.union(s2))


data = ["sdf", "sdf", "ertert", "wqeqwe"]
# Define the size of the population
POP_SIZE = 50
# Define the range of shingles and bands values
SHINGLE_RANGE = range(3, 15)
BANDS_RANGE = range(3, 15)
# Define the number of generations
NUM_GENERATIONS = 100


# Define the fitness function
def fitness(chromosome, data, threshold):
    num_shingles, num_bands = chromosome
    # Create shingles and bands
    shingles = create_shingles(data, num_shingles)
    bands = create_bands(shingles, num_bands)
    # Generate hash codes for the bands
    hash_codes = generate_hash_codes(bands)
    # Find similar data points using the hash codes
    similar_pairs = find_similar_pairs(data, hash_codes, threshold)
    # Calculate the accuracy of the search
    accuracy = len(similar_pairs) / len(data)
    # Return the fitness score, which is the accuracy of the search
    return accuracy


# Define the initialization function
def initialize_population():
    population = []
    for _ in range(POP_SIZE):
        # Create a chromosome by randomly selecting a shingle and bands value
        shingle = random.choice(SHINGLE_RANGE)
        bands = random.choice(BANDS_RANGE)
        chromosome = [shingle, bands]
        population.append(chromosome)
    return population


# Define the selection function
def select_parents(population):
    # Use tournament selection to select parents
    parents = []
    for _ in range(2):
        tournament = random.sample(population, 5)
        winner = max(tournament, key = lambda chromosome: fitness(chromosome, data))
        parents.append(winner)
    return parents


# Define the crossover function
def crossover(parents):
    # Use single-point crossover to create offspring
    crossover_point = random.randint(1, len(parents[0]) - 1)
    offspring = []
    offspring.append(parents[0][:crossover_point] + parents[1][crossover_point:])
    offspring.append(parents[1][:crossover_point] + parents[0][crossover_point:])
    return offspring


# Define the mutation function
def mutate(chromosome):
    # Mutate a single shingle or bands value with a small probability
    for i in range(len(chromosome)):
        if random.random() < 0.1:
            if i == 0:
                chromosome[i] = random.choice(SHINGLE_RANGE)
            else:
                chromosome[i] = random.choice(BANDS_RANGE)
    return chromosome


# Define the replacement function
def replace_population(population, offspring):
    # Use elitism to select the best chromosomes from the old population and the offspring
    combined_population = population + offspring
    scores = [fitness(chromosome, data) for chromosome in combined_population]
    sorted_population = [x for _, x in sorted(zip(scores, combined_population), reverse = True)]
    return sorted_population[:POP_SIZE]


# Generate data for testing the fitness function

# Initialize the population
population = initialize_population()
# Repeat the evolution process for a number of generations
for i in range(NUM_GENERATIONS):
    # Select parents
    parents = select_parents(population)
    # Create offspring through crossover and mutation
    offspring = [mutate(chromosome) for chromosome in crossover(parents)]
    # Replace the old population with the new offspring
    population = replace_population(population, offspring)
# Select the best solution
best_solution = max(population)
