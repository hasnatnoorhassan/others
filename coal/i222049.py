# Genetic Algorithm Cheat Sheet - Lab Ready

import random
import matplotlib.pyplot as plt

# ---------------------- Step 1: Initial Population ----------------------
def generate_population(size, gene_length):
    return [''.join(random.choice('01') for _ in range(gene_length)) for _ in range(size)]

# ---------------------- Step 2: Fitness Function ------------------------
def fitness(chromosome):
    return chromosome.count('1')  # Example: Maximize number of 1s

# ---------------------- Step 3: Selection -------------------------------

def roulette_wheel_selection(pop, fitnesses):
    total_fit = sum(fitnesses)
    pick = random.uniform(0, total_fit)
    current = 0
    for i, fit in enumerate(fitnesses):
        current += fit
        if current > pick:
            return pop[i]

def tournament_selection(pop, fitnesses, k=3):
    selected = random.sample(list(zip(pop, fitnesses)), k)
    return max(selected, key=lambda x: x[1])[0]

def rank_selection(pop, fitnesses):
    sorted_pop = sorted(zip(pop, fitnesses), key=lambda x: x[1])
    ranks = list(range(1, len(pop)+1))
    total = sum(ranks)
    pick = random.randint(1, total)
    current = 0
    for (ind, fit), rank in zip(sorted_pop, ranks):
        current += rank
        if current >= pick:
            return ind

# ---------------------- Step 4: Crossover -------------------------------

def single_point_crossover(p1, p2):
    point = random.randint(1, len(p1)-1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def two_point_crossover(p1, p2):
    pt1, pt2 = sorted(random.sample(range(len(p1)), 2))
    return (p1[:pt1] + p2[pt1:pt2] + p1[pt2:],
            p2[:pt1] + p1[pt1:pt2] + p2[pt2:])

def uniform_crossover(p1, p2):
    child1, child2 = '', ''
    for i in range(len(p1)):
        if random.random() < 0.5:
            child1 += p1[i]
            child2 += p2[i]
        else:
            child1 += p2[i]
            child2 += p1[i]
    return child1, child2

# ---------------------- Step 5: Mutation --------------------------------

def bit_flip_mutation(chromosome, mutation_rate=0.01):
    return ''.join(
        bit if random.random() > mutation_rate else str(1 - int(bit))
        for bit in chromosome
    )

def swap_mutation(chromosome):
    i, j = random.sample(range(len(chromosome)), 2)
    lst = list(chromosome)
    lst[i], lst[j] = lst[j], lst[i]
    return ''.join(lst)

# ---------------------- Step 6: Termination -----------------------------

def is_termination(pop, generations, max_generations, threshold):
    best_fit = max(fitness(ind) for ind in pop)
    return generations >= max_generations or best_fit >= threshold

# ---------------------- Genetic Algorithm Loop --------------------------

def genetic_algorithm():
    population = generate_population(10, 8)
    max_generations = 100
    threshold = 8
    generations = 0
    best_fitness_over_time = []

    while not is_termination(population, generations, max_generations, threshold):
        fitnesses = [fitness(ind) for ind in population]
        best_fitness_over_time.append(max(fitnesses))
        new_population = []
        while len(new_population) < len(population):
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            c1, c2 = single_point_crossover(p1, p2)
            new_population.extend([
                bit_flip_mutation(c1),
                bit_flip_mutation(c2)
            ])
        population = new_population
        generations += 1

    print("Best Individual:", max(population, key=fitness))
    print("Generations:", generations)

    # Plotting
    plt.plot(best_fitness_over_time)
    plt.title("Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.show()

# ---------------------- Run the Algorithm -------------------------------

genetic_algorithm()