import random
import numpy as np
import functools

import co_functions as cf
import utils
import matplotlib.pyplot as plt

DIMENSION = 10 # dimension of the problems
POP_SIZE = 100 # population size
MAX_GEN = 100 # maximum number of generations
CX_PROB = 0.8 # crossover probability
MUT_PROB = 0.2 # mutation probability
MUT_STEP = 0.1 # size of the mutation steps
REPEATS = 10 # number of runs of algorithm (should be at least 10)
OUT_DIR = 'continuous' # output directory for logs
#EXP_ID = 'lamarck' # the ID of this experiment (used to create log names)
EXP_ID = 'test_Baldwin' # the ID of this experiment (used to create log names)
learning_rate = 0.01
lifetime = 5

# creates the individual
def create_ind(ind_len):
    return np.random.uniform(-5, 5, size=(ind_len,))
    #return np.append(np.random.uniform(-5, 5, size=(ind_len,)),[np.random.uniform(0, 2), np.random.uniform(0, 1)])
# creates the population using the create individual function
def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

# the tournament selection (roulette wheell would not work, because we can have 
# negative fitness)
def tournament_selection(pop, fits, k):
    selected = []
    for i in range(k):
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

# gaussian mutation - we need a class because we want to change the step
# size of the mutation adaptively
class Mutation:

    def __init__(self, step_size):
        self.step_size = step_size

    def __call__(self, ind):
        return ind + self.step_size*np.random.normal(size=ind.shape)

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
'''
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, mutate_ind, *, map_fn=map, log=None):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]

        mating_pool = mate_sel(pop, fits, POP_SIZE)
        offspring = mate(mating_pool, operators)
        pop = offspring[:]

    return pop


def larmarckism(parents, parents_fitness, mutate, fitness, map_fn):
    # improve the parents with a mutation
    betterParents = mutate(parents)
    fits_objs = list(map_fn(fitness, betterParents))
    betterParentsFit = [f.fitness for f in fits_objs]

    new_parent_pop = []

    for i in range(len(parents)):
        if parents_fitness[i] > betterParentsFit[i]:
            new_parent_pop.append(parents[i])
        else:
            new_parent_pop.append(betterParents[i])

    return new_parent_pop

#def baldwinism(parents, parents_fitness, mutate, fitness, map_fn):
    
    

class Gradient_descent:
    def __init__(self, fit_func, life_duration=lifetime):
        self.fit_func = fit_func
        #self.learning_method = learning_method
        self.life_duration = life_duration
        
    def mutation_gradient(self, ind):
        ind -= learning_rate * cf.numerical_derivative(self.fit_func, ind)
        return ind

def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, mutate_ind, *, map_fn=map, log=None):
    evo = Gradient_descent(fit_func=fitness)
    evals = 0
    for G in range(max_gen):
        
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]
        for _ in range (lifetime):
            pop = list(map_fn(evo.mutation_gradient, pop))

        # adaptive stepsize
        if G % 12 == 0:
            mutate_ind.step_size = mutate_ind.step_size*0.995

        mating_pool = mate_sel(pop, fits, POP_SIZE)
        mating_pool = larmarckism(mating_pool, fits, operators[1], fitness, map_fn)
        offspring = mate(mating_pool, operators[0]) # lamarkism offpring, crossover here, use mutate fn above

        # original offspring function
        # offspring = mate(mating_pool, operators)

        # run this without mating_pool and offspring for differntial evolution
        #offspring = diff_evolution(pop, fits, fitness, map_fn)
        pop = offspring[:]

    return pop

'''

def baldwinism(parents, parents_fitness, mutate, fitness, map_fn):
    betterParents = mutate(parents)
    fits_objs = list(map_fn(fitness, betterParents))
    betterParentsFit = [f.fitness for f in fits_objs]

    new_parent_pop = []

    for i in range(len(parents)):
        if parents_fitness[i] > betterParentsFit[i]:
            new_parent_pop.append(parents[i])
        else:
            new_parent_pop.append(betterParents[i])

    return betterParents
    
    
class Gradient_descent:
    def __init__(self, fit_func, life_duration=lifetime):
        self.fit_func = fit_func
        #self.learning_method = learning_method
        self.life_duration = life_duration
        
    def gradient(self, ind):
        ind -= learning_rate * cf.numerical_derivative(self.fit_func, ind)
        return ind

def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, mutate_ind, *, map_fn=map, log=None):
    evo = Gradient_descent(fit_func=fitness)
    
    evals = 0
    fits_objs = list(map_fn(evo.fit_func, pop))
    if log:
        log.add_gen(fits_objs, evals)
    initial_fit = [f.fitness for f in fits_objs]
    for G in range(max_gen):
        
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]
        for _ in range (lifetime):
            pop = list(map_fn(evo.gradient, pop))

        # adaptive stepsize
        if G % 12 == 0:
            mutate_ind.step_size = mutate_ind.step_size*0.995

        mating_pool = mate_sel(pop, [initial_fit - fits for initial_fit, fits in zip(initial_fit, fits)], POP_SIZE)
        #mating_pool = baldwinism(mating_pool, fits, initial_fit, fitness, map_fn)
        intial_fit = fits
        offspring = mate(mating_pool, operators) # lamarkism offpring, crossover here, use mutate fn above

        # original offspring function
        # offspring = mate(mating_pool, operators)

        # run this without mating_pool and offspring for differntial evolution
        #offspring = diff_evolution(pop, fits, fitness, map_fn)
        pop = offspring[:]

    return pop

    
if __name__ == '__main__':

    # use `functool.partial` to create fix some arguments of the functions 
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, ind_len=DIMENSION)
    # we will run the experiment on a number of different functions
    fit_generators = [cf.make_f08_rosenbrock]
    '''
    [cf.make_f01_sphere]
    cf.make_f02_ellipsoidal,
    cf.make_f06_attractive_sector,
    cf.make_f08_rosenbrock,
    cf.make_f10_rotated_ellipsoidal]
    '''
    #fit_names = ['f01', 'f02', 'f06', 'f08', 'f10']
    fit_names = ['f08'] 

    for fit_gen, fit_name in zip(fit_generators, fit_names):
        fit = fit_gen(DIMENSION)
        mutate_ind = Mutation(step_size=MUT_STEP)
        xover = functools.partial(crossover, cross=one_pt_cross, cx_prob=CX_PROB)
        mut = functools.partial(mutation, mut_prob=MUT_PROB, mutate=mutate_ind)

        # run the algorithm `REPEATS` times and remember the best solutions from 
        # last generations
    
        best_inds = []
        for run in range(REPEATS):
            # initialize the log structure
            log = utils.Log(OUT_DIR, EXP_ID + '.' + fit_name , run, 
                            write_immediately=True, print_frequency=5)
            # create population
            pop = create_pop(POP_SIZE, cr_ind)
            # run evolution - notice we use the pool.map as the map_fn
            pop = evolutionary_algorithm(pop, MAX_GEN, fit, [xover, mut], tournament_selection, mutate_ind, map_fn=map, log=log)
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
    plt.legend()
    plt.yscale('log')
    plt.show()