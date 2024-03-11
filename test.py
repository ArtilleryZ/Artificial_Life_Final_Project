import numpy as np
import mujoco
import mujoco_viewer
import time
import final_xml_generator

# Genetic Algorithm Components
def generate_initial_population(pop_size, param_ranges):
    population = np.random.uniform(param_ranges['min'], param_ranges['max'], (pop_size, len(param_ranges['min'])))
    return population

def simulate_robot(individual, iter):
    filename = f"xml/final{iter}.xml"
    final_xml_generator.generate_robot_xml(filename, *individual)
    m = mujoco.MjModel.from_xml_path(filename)
    d = mujoco.MjData(m)
    viewer = mujoco_viewer.MujocoViewer(m, d)
    height = []

    for _ in range(1500):
        height.append(d.sensordata[2])
        mujoco.mj_step(m, d)
    viewer.close()

    return max(height)

def evaluate_population(population):
    fitness = [simulate_robot(individual, i) for i, individual in enumerate(population)]
    return np.array(fitness)

def select_parents(population, fitness, num_parents):
    parents_idx = np.argsort(-fitness)[:num_parents]
    return population[parents_idx]

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1]//2)

    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

    return offspring

def mutate(offspring, param_ranges):
    for idx in range(offspring.shape[0]):
        mutation_idx = np.random.randint(0, len(param_ranges['min']))
        random_value = np.random.uniform(-1.0, 1.0, 1) * 0.05 * (param_ranges['max'][mutation_idx] - param_ranges['min'][mutation_idx])
        offspring[idx, mutation_idx] = offspring[idx, mutation_idx] + random_value
        offspring[idx, mutation_idx] = np.clip(offspring[idx, mutation_idx], param_ranges['min'][mutation_idx], param_ranges['max'][mutation_idx])
    return offspring

# Parameters
param_ranges = {'min': np.array([0.18, 0.13, 0.08, 0.3, 0.02, 0.02, 0.3, 0.02, 0.02]), 
                'max': np.array([0.22, 0.17, 0.12, 0.5, 0.15, 0.15, 0.5, 0.15, 0.15])}
pop_size = 10
num_generations = 5
num_parents_mating = 4

# Genetic Algorithm
population = generate_initial_population(pop_size, param_ranges)
for generation in range(num_generations):
    print(f"Generation {generation}")
    fitness = evaluate_population(population)
    parents = select_parents(population, fitness, num_parents_mating)
    offspring_crossover = crossover(parents, (pop_size - parents.shape[0], parents.shape[1]))
    offspring_mutation = mutate(offspring_crossover, param_ranges)
    population[:parents.shape[0], :] = parents
    population[parents.shape[0]:, :] = offspring_mutation

# Final Evaluation
final_fitness = evaluate_population(population)
best_index = np.argmax(final_fitness)
best_solution = population[best_index]
print("Best Solution: ", best_solution)
print("Best Solution Fitness: ", final_fitness[best_index])
