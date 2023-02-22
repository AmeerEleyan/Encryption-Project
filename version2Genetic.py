# Created by Ameer Eleyan
# at 2/16/2023 9:40 PM

import random
import numpy as np
from collections import defaultdict
import LSH


def create_shingles(corpus, shingle_size):
    shingles = []
    for d in corpus:
        shingles.append(LSH.shingle(data = list(d)[0], shingle_size = shingle_size))
    return shingles


def create_bands(shingles, num_bands):
    bands = []
    for s in shingles:
        b = []
        for i in range(num_bands):
            start_index = i * len(s) // num_bands
            end_index = (i + 1) * len(s) // num_bands
            band = frozenset(list(s)[start_index:end_index])
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


def find_similar_pairs(corpus, hash_codes, threshold):
    similar_pairs = set()
    buckets = defaultdict(list)
    for i, hc in enumerate(hash_codes):
        for j, h in enumerate(hc):
            buckets[h].append(i)
    for bucket in buckets.values():
        if len(bucket) > 1:
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    d1 = corpus[bucket[i]]
                    d2 = corpus[bucket[j]]
                    if LSH.jaccard_similarity(d1, d2) >= threshold:
                        similar_pairs.add((bucket[i], bucket[j]))
    return similar_pairs


# Define the size of the population
POP_SIZE = 10
# Define the range of shingles and bands values
SHINGLE_RANGE = range(2, 50)
BANDS_RANGE = range(2, 50)
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


def select_parents(population, data, threshold):
    # Use tournament selection to select parents
    parents = []
    for i in range(2):
        tournament = random.sample(population, 5)
        tournament_fitness = [fitness(chromosome, data, threshold) for chromosome in tournament]
        winner = tournament[np.argmax(tournament_fitness)]
        parents.append(winner)
    return parents


# Define the crossover function
def crossover(parents):
    # Perform single-point crossover to create two new children
    crossover_point = random.randint(0, len(parents[0]))
    child1 = parents[0][:crossover_point] + parents[1][crossover_point:]
    child2 = parents[1][:crossover_point] + parents[0][crossover_point:]
    return [child1, child2]


# Define the mutation function
def mutate(chromosome):
    # Mutate a random gene in the chromosome
    gene_index = random.randint(0, len(chromosome) - 1)
    if gene_index == 0:
        # Mutate the shingle value
        chromosome[gene_index] = random.choice(SHINGLE_RANGE)
    else:
        # Mutate the bands value
        chromosome[gene_index] = random.choice(BANDS_RANGE)
    return chromosome


# Define the genetic algorithm function
def genetic_algorithm(data, threshold):
    # Initialize the population

    global best_individual, best_fitness
    population = initialize_population()
    # Iterate through generations
    for i in range(NUM_GENERATIONS):
        # Select parents and create children
        children = []
        for j in range(POP_SIZE // 2):
            parents = select_parents(population, data, threshold)
            child1, child2 = crossover(parents)
            # Mutate the children
            child1 = mutate(child1)
            child2 = mutate(child2)
            children.extend([child1, child2])
        # Evaluate the fitness of the children
        fitness_scores = [fitness(chromosome, data, threshold) for chromosome in children]
        # Select the top individuals to survive
        population = [x for _, x in sorted(zip(fitness_scores, children), reverse = True)][:POP_SIZE]
        # Print the best individual and its fitness score
        best_individual = population[0]
        best_fitness = fitness(best_individual, data, threshold)
        # print(f"Generation {i + 1}: Best individual {best_individual}, fitness score {best_fitness}")
    # Print the shingle size, bands, and min hash values for the best individual
    print("Best individual:")
    print("  Shingle size:", best_individual[0])
    print("  Bands:", best_individual[1])
    shingles = create_shingles(data, best_individual[0])
    bands = create_bands(shingles, best_individual[1])
    hash_codes = generate_hash_codes(bands)
    print("  Min hash values:", len(hash_codes))
    return best_individual, best_fitness


data = [
    {'Ameer fdg dfg ert gao sdipfew feepwre'},
]
threshold = 0.5
genetic_algorithm(data, threshold)
