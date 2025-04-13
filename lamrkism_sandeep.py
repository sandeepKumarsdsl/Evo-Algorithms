import random
import numpy as np
import functools
import co_functions as cf
import utils
from co_functions import numerical_derivative

import matplotlib.pyplot as plt


DIMENSION = 10  # dimension of the problems
POP_SIZE = 100  # population size
MAX_GEN = 100  # maximum number of generations
CX_PROB = 0.8  # crossover probability
MUT_PROB = 0.2  # mutation probability
MUT_STEP = 0.01  # size of the mutation steps
REPEATS = 10  # number of runs of algorithm (should be at least 10)
OUT_DIR = 'continuous'  # output directory for logs
EXP_ID = 'lamarkism_high_lr_version_new'  # the ID of this experiment (used to create log names)



#=================
learning_rate = 0.1
lifetime = 5
gradient_descend = lambda fit_func, ind: ind - learning_rate * numerical_derivative(fit_func, ind)
#=================



class lamark_evol:
    def __init__(self, fit_func, learning_method=gradient_descend, life_duration=lifetime):
        self.fit_func = fit_func
        self.learning_method = learning_method
        self.life_duration = life_duration
        self.fits_born = None

    def fit(self, pop, evals, map_fn, log):
        fits_objs = list(map_fn(self.fit_func, pop))
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        return (fits)

    def ind_adapt(self, ind, life_t=None, learning_method=None):
       #print(type(ind))
        if learning_method is None:
            learning_method = self.learning_method
        if life_t is None:
            life_t = self.life_duration
        for _ in range(life_t):
            ind = learning_method(self.fit_func, ind)    
        return (ind)

    def selection(self, pop, fits_dead, mate_sel):
        mating_pool = mate_sel(pop, fits_dead, POP_SIZE)
        self.fits_born = fits_dead
        return (mating_pool)



def evol_algo(pop, max_gen, fitness, operators, mate_sel, mutate_ind, *, map_fn=map, log=None):
    evolution = lamark_evol(fit_func=fitness)
    evals = 0
    evolution.fits_born = evolution.fit(pop, evals, map_fn, log=None)
    
    for G in range(max_gen):
        
        pop = list(map_fn(evolution.ind_adapt, pop))
        #print(pop)
        evals += len(pop)
        fits_dead = evolution.fit(pop, evals, map_fn, log=log)
        mating_pool = evolution.selection(pop, fits_dead, mate_sel, )
        offspring = mate(mating_pool, operators)
        pop = offspring[:]
        #print(type(pop))
    return pop


def create_ind(ind_len):
    return np.random.uniform(-5, 5, size=(ind_len,))


def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]


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


def one_pt_cross(p1, p2):
    point = random.randrange(1, len(p1))
    o1 = np.append(p1[:point], p2[point:])
    o2 = np.append(p2[:point], p1[point:])
    return (o1, o2)


class Mutation:
    def __init__(self, step_size):
        self.step_size = step_size
    def __call__(self, ind):
        return ind + self.step_size * np.random.normal(size=ind.shape)


def mate(pop, operators):
    for o in operators:
        pop = o(pop)
    return pop


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



def mutation(pop, mutate, mut_prob):
    return [mutate(p) if random.random() < mut_prob else p[:] for p in pop]


def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, mutate_ind, *, map_fn=map, log=None):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]
        best_fits = min(fits)
        mating_pool = mate_sel(pop, fits, POP_SIZE)
        offspring = mate(mating_pool, operators)
        pop = offspring[:]

    return pop


if __name__ == '__main__':
    # use `functool.partial` to create fix some arguments of the functions
    # and create functions with required signatures
    cr_ind = functools.partial(create_ind, ind_len=DIMENSION)
    # we will run the experiment on a number of different functions
    fit_generators = [cf.make_f01_sphere]
    '''
    [cf.make_f01_sphere]
    cf.make_f02_ellipsoidal,
    cf.make_f06_attractive_sector,
    cf.make_f08_rosenbrock,
    cf.make_f10_rotated_ellipsoidal]
    '''
    #fit_names = ['f01', 'f02', 'f06', 'f08', 'f10']
    fit_names = ['f01'] 
    for fit_gen, fit_name in zip(fit_generators, fit_names):
        fit = fit_gen(DIMENSION)
        mutate_ind = Mutation(step_size=MUT_STEP)
        xover = functools.partial(crossover, cross=one_pt_cross, cx_prob=CX_PROB)
        mut = functools.partial(mutation, mut_prob=MUT_PROB, mutate=mutate_ind)
        best_inds = []

        for run in range(REPEATS):
            # initialize the log structure
            log = utils.Log(OUT_DIR, EXP_ID + '.' + fit_name, run,
                            write_immediately=True, print_frequency=5)
            pop = create_pop(POP_SIZE, cr_ind)
            pop = evol_algo(pop, MAX_GEN, fit, [xover, mut], tournament_selection, None, map_fn=map,
                                          log=log)
            bi = max(pop, key=fit)
            best_inds.append(bi)

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