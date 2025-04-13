import random
import numpy as np
import functools
import matplotlib.pyplot as plt
import co_functions as cf
import utils

DIMENSION = 10  # dimension of the problems
POP_SIZE = 200  # population size
MAX_GEN = 500  # maximum number of generations
CX_PROB = 0.0  # crossover probability
C_CX_PROB = 0.0  # intelligent arithmetic crossover probability
R_CX_PROB = 0.0  # intelligent arithmetic crossover probability
MUT_PROB = 0.0  # mutation probability
MUT_STEP = 0.0  # size of the mutation steps
REPEATS = 10  # number of runs of algorithm (should be at least 10)
DIFF_F = 0.5  # Differential F param
DIFF_C = 0.9  # Differential C param
DIFF_X = 2  # Number of randomly chosen neighbours for the difference
OUT_DIR = 'continuous'  # output directory for logs
# the ID of this experiment (used to create log names)
EXP_ID = 'diff-threesome-adaptive-boosted'

# creates the individual
def create_ind(ind_len):
    return np.append(np.random.uniform(-5, 5, size=(ind_len,)), [np.random.uniform(0, 2), np.random.uniform(0, 1)])

# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

# the tournament selection (roulette wheell would not work, because we can have
# negative fitness)


def tournament_selection(pop, fits, k):
    selected = []
    for _ in range(k):
        p1 = random.randrange(0, len(pop))
        p2 = random.randrange(0, len(pop))
        if fits[p1] > fits[p2]:
            selected.append(np.copy(pop[p1]))
        else:
            selected.append(np.copy(pop[p2]))

    return selected

# implements the one-point crossover of two individuals


def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1))
    o1 = np.append(p1[:point], p2[point:])
    o2 = np.append(p2[:point], p1[point:])
    return o1, o2


def coin_arithmetic_cross(p1, p2):
    coin = np.random.uniform(size=len(p1))
    a = 0 + coin
    b = 1 - coin
    return a * p1 + b * p2, b * p1 + a * p2


def random_arithmetic_cross(p1, p2):
    w = random.random()

    return w*p1+(1-w)*p2, (1-w)*p1+w*p2


# gaussian mutation - we need a class because we want to change the step
# size of the mutation adaptively


class Mutation:

    def __init__(self, step_size):
        self.step_size = step_size

    def __call__(self, ind):
        return ind + self.step_size*np.random.normal(size=ind.shape)


class DiffMutation:

    def __init__(self, population, fitness):
        self.population = population
        self.fitness = fitness

    def __call__(self, ind):
        partners = np.zeros((DIFF_X+1, len(ind)))
        partners[0] = ind
        for _ in range(DIFF_X):
            r = np.random.randint(0, len(self.population))
            indx = 0
            while any([np.array_equal(self.population[r], x) for x in partners]):
                r = np.random.randint(0, len(self.population))
                indx += 1
                if indx > 100:
                    return ind

            partners[_+1] = self.population[r]

        donor = partners[1] + ind[-2] * \
            (np.sum(partners[2::2]) - np.sum(partners[3::2]))
        offspring = [donor[i] if np.random.rand() < ind[-1] else ind[i]
                     for i in range(len(ind))]

        ind_fit = self.fitness(ind).fitness
        off_fit = self.fitness(offspring).fitness

        return ind if ind_fit > off_fit else offspring

# applies a list of genetic operators (functions with 1 argument - population)
# to the population


def mate(pop, operators):
    for o in operators:
        pop = o(pop)
    return pop

# applies the cross function (implementing the crossover of two individuals)
# to the whole population (with probability cx_prob)


def crossover(pop, cross, cx_prob):
    off = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):
        if random.random() < cx_prob:
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = p1[:], p2[:]
        off.append(o1)
        off.append(o2)
    return off

# applies the mutate function (implementing the mutation of a single individual)
# to the whole population with probability mut_prob)


def mutation(pop, mutate, mut_prob):
    return [mutate(p) if random.random() < mut_prob else p[:] for p in pop]

# implements the evolutionary algorithm
# arguments:
#   pop_size  - the initial population
#   max_gen   - maximum number of generation
#   fitness   - fitness function (takes individual as argument and returns
#               FitObjPair)
#   operators - list of genetic operators (functions with one arguments -
#               population; returning a population)
#   mate_sel  - mating selection (funtion with three arguments - population,
#               fitness values, number of individuals to select; returning the
#               selected population)
#   mutate_ind - reference to the class to mutate an individual - can be used to
#               change the mutation step adaptively
#   map_fn    - function to use to map fitness evaluation over the whole
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run


def evolutionary_algorithm(pop, max_gen, fitness, mate_sel, mutate_ind, *, map_fn=map, log=None):
    evals = 0
    for _ in range(max_gen):
        mutate_ind.population = pop
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        _ = [f.objective for f in fits_objs]

        # mating_pool = pop  # mate_sel(pop, fits, POP_SIZE)mate(mating_pool, [])
        offspring = mutation(pop, mutate_ind, 1.0)
        pop = offspring[:]

    return pop


def individual_fitness(ind, func, dim):
    return func(ind[:dim])

if __name__ == '__main__':
    # use `functool.partial` to create fix some arguments of the functions
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, ind_len=DIMENSION)
    # we will run the experiment on a number of different functions
    fit_generators = [cf.make_f01_sphere,
                      cf.make_f02_ellipsoidal,
                      cf.make_f06_attractive_sector,
                      cf.make_f08_rosenbrock,
                      cf.make_f10_rotated_ellipsoidal]
    fit_names = ['f01', 'f02', 'f06', 'f08', 'f10']

    for fit_gen, fit_name in zip(fit_generators, fit_names):
        fit = fit_gen(DIMENSION)
        fit = functools.partial(individual_fitness, func=fit, dim=DIMENSION)

        mutate_ind = DiffMutation(population=[], fitness=fit)

        # xover = functools.partial(
        #     crossover, cross=one_pt_cross, cx_prob=CX_PROB)
        # rxover = functools.partial(
        #     crossover, cross=random_arithmetic_cross, cx_prob=R_CX_PROB)
        # cxover = functools.partial(
        #     crossover, cross=coin_arithmetic_cross, cx_prob=C_CX_PROB)
        # mut = functools.partial(mutation, mut_prob=1.0, mutate=mutate_ind)

        # run the algorithm `REPEATS` times and remember the best solutions from
        # last generations

        best_inds = []
        for run in range(REPEATS):
            # initialize the log structure
            log = utils.Log(OUT_DIR, EXP_ID + '.' + fit_name, run,
                            write_immediately=True, print_frequency=5)
            # create population
            pop = create_pop(POP_SIZE, cr_ind)
            # run evolution - notice we use the pool.map as the map_fn
            pop = evolutionary_algorithm(
                pop, MAX_GEN, fit, tournament_selection, mutate_ind, map_fn=map, log=log)
            # remember the best individual from last generation, save it to file
            bi = max(pop, key=fit)
            best_inds.append(bi)

            # if we used write_immediately = False, we would need to save the
            # files now
            # log.write_files()

        # print an overview of the best individuals from each run
        for i, bi in enumerate(best_inds):
            print(f'Run {i}: objective = {fit(bi).objective}')

        # write summary logs for the whole experiment
        utils.summarize_experiment(OUT_DIR, EXP_ID + '.' + fit_name)
        evals, lower, mean, upper = utils.get_plot_data(OUT_DIR, EXP_ID+ '.' + fit_name)
        utils.plot_experiment(evals, lower, mean, upper, legend_name = fit_name)
    
        '''evals, lower, mean, upper = utils.get_plot_data(OUT_DIR, EXP_ID+ '.' + fit_name)
        plt.figure(figsize=(12, 8))
        utils.plot_experiment(evals, lower, mean, upper, legend_name = 'Default settings')
        plt.legend()
        plt.show()'''
    plt.xlabel('Fitness')
    plt.ylabel('Growth')
    plt.title('Diff_eval')
    plt.legend()
    plt.yscale('log')
    plt.show()

