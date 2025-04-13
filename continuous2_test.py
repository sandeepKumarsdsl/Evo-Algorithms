import random
import numpy as np
import functools
import co_functions as cf
import utils


# creates the individual
class PopulationCreator:
    def __init__(self, ind_len, gammas=False, alphas=False):
        self.ind_len = ind_len
        self.gammas = gammas
        self.alphas = alphas
        self.default_sigma = 0.5
        self.default_alpha = 0.0

    def init_population(self, pop_len):
        return [self.create_individual_with_gammas_and_alphas() for _ in range(pop_len)]

    def create_x(self):
        return np.random.uniform(-5, 5, size=(self.ind_len,))

    def create_individual_with_gammas_and_alphas(self):
        return np.concatenate((np.expand_dims(self.create_x(), axis=0), np.expand_dims(self.create_gammas(), axis=0),  self.create_alphas()), axis=0)

    def create_gammas(self):
        return np.full(self.ind_len, self.default_sigma)

    def create_alphas(self):
        return np.full((self.ind_len, self.ind_len), self.default_alpha)


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
def one_pt_cross(p1, p2):  # todo -- 1 point crossover implementation
    point = random.randrange(1, p1.shape[1])
    o1 = np.append(p1[:, :point], p2[:, point:], axis=1)
    o2 = np.append(p2[:, :point], p1[:, point:], axis=1)
    return o1, o2


def weighted_cross(p1, p2):
    w = np.random.random()
    o1 = w*p1 + (1-w)*p2
    o2 = w*p1 - (1-w)*p2
    return o1, o2


def no_cross(p1, p2):
    return p1, p2


def SBX_cross(p1, p2):
    beta = np.random.normal(loc=1, scale=0.2)
    o1 = 1/2*(p1+p2) + beta*1/2*(p1-p2)
    o2 = 1/2*(p1+p2) - beta*1/2*(p1-p2)
    return o1, o2

# applies the cross function (implementing the crossover of two individuals)
# to the whole population (with probability cx_prob)
def crossover(pop, cross, cx_prob):  # todo -- this is crossover wrapper .. only 2-parent -> 2-child crossover ...
    off = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):  # 0 2 4 ;;;; 1 3 5
        if random.random() < cx_prob:
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = p1[:], p2[:]
        off.append(o1)
        off.append(o2)
    return off

class MutationInterface:
    def at_episode_end(self, fits_before, fits_after):
        pass


# gaussian mutation - we need a class because we want to change the step
# size of the mutation adaptively
class MutationConstant(MutationInterface):

    def __init__(self, step_size, tmp=None):
        self.step_size = step_size

    def __call__(self, ind):
        return ind + self.step_size * np.random.normal(size=ind.shape)


class MutationLinearDecay(MutationInterface):
    def __init__(self, step_size):
        self.scale = step_size  # sigma == standard deviation == scale
        self.episode_num = 0
        self.n = 1  # decay every n-th episode
        assert self.n > 0
        self.decay = 0.995

    def __call__(self, ind):
        return ind + np.random.normal(loc=0, scale=self.scale, size=ind.shape)

    # Decay every n-th episode
    def at_episode_end(self, fits_before, fits_after):
        self.episode_num += 1
        if self.episode_num % self.n == 0:
            self.scale *= self.decay


class MutationScalable(MutationInterface):
    def __init__(self, step_size):
        self.sigma = None
        self.default_sigma = step_size

    def __call__(self, ind):
        n = ind[0, :].shape[0]

        if self.sigma is None:
            self.sigma = np.full(n, self.default_sigma)

        # ja bych potreboval nagenerovat asi tak 1000 ruznych variant sigmy ....
        # s kazdou sigmou project celou population ..... (nagenerovat 10x tolik potomku co je potreba)
        #
        t = 1 /np.power(n, 1/2)
        t_i = 1 /np.power(n, 1/4)

        # sigma [elementwise] e^{ N(0, t_i) } *  e^N(0,t)
        a = np.exp(np.random.normal(loc=0, scale=t_i, size=self.sigma.shape[0]))
        b = np.exp(t)
        mutated_sigma = np.multiply(self.sigma, a) * b

        # x + sigma * N(0,1)
        return ind + np.random.normal(loc=0, scale=mutated_sigma, size=ind.shape)


class MutationScalableIndividual(MutationInterface):
    def __init__(self, step_size):
        pass

    def __call__(self, ind):
        x = ind[0, :]
        sigma = ind[1, :]
        alphas = ind[2:, :]
        n = ind[0, :].shape[0]
        t = 1/np.power(n, 1/2)
        t_i = 1/np.power(n, 1/4)
        t = 1
        t_i = 1

        # sigma * e^{t_i N(0,1)} * t
        mutated_sigma = np.multiply(sigma, t_i * np.exp(np.random.normal(loc=0, scale=1, size=sigma.shape[0]))) * t

        # x + sigma * N(0,1)
        mutated_x = x + np.random.normal(loc=0, scale=mutated_sigma, size=x.shape[0])
        ind_new = np.concatenate((np.expand_dims(mutated_x, axis=0),np.expand_dims(mutated_sigma, axis=0), alphas), axis=0)

        assert ind.shape == ind_new.shape

        return ind_new


class RotatableIndividual(MutationInterface):
    def __init__(self, step_size):
        pass

    def __call__(self, ind):
        x = ind[0, :]
        sigma = ind[1, :]
        alphas = ind[2:, :]

        n = ind[0, :].shape[0]
        t = 1/np.power(n, 1/2)
        t_i = 1/np.power(n, 1/4)
        t = 1
        t_i = 1
        b = 5

        # sigma * e^{t_i N(0,1)} * t
        mutated_sigma = np.multiply(sigma, t_i * np.exp(np.random.normal(loc=0, scale=1, size=sigma.shape[0]))) * t
        mutated_alpha = alphas + b * np.random.normal(loc=0, scale=1, size=alphas.shape)
        # print(mutated_alpha.shape[0])
        # print(mutated_alpha.shape[1])

        for i in range(mutated_alpha.shape[0]):
            for j in range(mutated_alpha.shape[1]):
                alp_i, alp_j = i, j
                if j > i:
                    continue

                # xx = 0
                while np.abs(mutated_alpha[alp_i, alp_j]) > np.pi:
                    # print("correction: {} -> ".format(mutated_alpha[alp_i, alp_j]), end="")
                    mutated_alpha[alp_i, alp_j] -= 2*np.pi * np.sign( mutated_alpha[alp_i, alp_j])
                    # print("{}".format(mutated_alpha[alp_i, alp_j]))
                    # xx + =1
                    # print("---{}".format(xx))
                mutated_alpha[alp_j, alp_i] = mutated_alpha[alp_i, alp_j]

        C = np.power(mutated_sigma,2) * np.eye(sigma.shape[0])
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                if i != j:
                    if i > j:
                        continue
                    # for the alpha, use the triangle only ...
                    # alp_i, alp_j = i, j
                    # if j > i:
                    #     alp_i, alp_j = j, i
                    C[i, j] = 1/2 * np.abs((np.power(mutated_sigma[i], 2) - np.power(mutated_sigma[j], 2)))*np.tan(2* mutated_alpha[i, j])
                    C[j, i] = C[i, j]

        # def check_symmetric(a, rtol=1e-05, atol=1e-08):
        #     return np.allclose(a, a.T, rtol=rtol, atol=atol)
        # print("Symmetric? {}".format(check_symmetric(C)))

        mutated_x = x + np.random.multivariate_normal(mean=[0]*C.shape[0], cov=C)
        ind_new = np.concatenate((np.expand_dims(mutated_x, axis=0),np.expand_dims(mutated_sigma, axis=0), mutated_alpha), axis=0)

        assert ind.shape == ind_new.shape

        return ind_new

class MutationRule20(MutationInterface):
    def __init__(self, step_size):
        self.scale = step_size  # sigma == standard deviation == scale
        self.avg_episode_success = []
        self.n = 1  # check the average mutation success every n-th episode
        assert self.n > 0
        self.rule = 0.2

    def __call__(self, ind):
        return ind + np.random.normal(loc=0, scale=self.scale, size=ind.shape)

    def at_episode_end(self, fits_before, fits_after):
        assert len(fits_before) == len(fits_after)
        num_improves = sum([1 if new_fit > old_fit else 0 for new_fit, old_fit in zip(fits_after, fits_before)])
        avg_suc = num_improves / len(fits_before)
        # print(avg_suc)
        self.avg_episode_success.append(avg_suc)

        # check every n-th episode
        if len(self.avg_episode_success) % self.n == 0:
            # avg over last n episode ...
            avg = sum(self.avg_episode_success[-self.n:]) / self.n
            if avg > self.rule:
                self.scale = self.scale * np.exp(1-1/5)

            elif avg < self.rule:
                self.scale = self.scale * np.exp(0-1/5)



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

def differential_evo(population, max_gen, fitness_f, F=0.8, CR=0.9, F_decay = 1, CR_decay = 1, from4=False, log=None):
    evals=0
    
    POP_LLEN = len(population)
    for G in range(max_gen):
        F = F*F_decay
        CR = CR*CR_decay
        fits_objs = list(map(fitness_f, [x[0, :] for x in population]))
        evals += len(population)
        if log:
            log.add_gen(fits_objs, evals)
        old_fit = [f.fitness for f in fits_objs]  # fitness function

        new_population = []
        for individual in population:

            # Mutation ....

            # kombinuje 2 (not 4)
            if not from4:
                i1 = population[np.random.randint(low=0, high=len(population))] # individual 1
                i2 = population[np.random.randint(low=0, high=len(population))] # individual 2
                i_mut = individual + F*(i1 - i2)  # posunu ve smeru rozdilu

            # kombinuji 4 dohromady...
            else:
                a = population[np.random.randint(low=0, high=len(population))]
                b = population[np.random.randint(low=0, high=len(population))]
                c = population[np.random.randint(low=0, high=len(population))]
                d = population[np.random.randint(low=0, high=len(population))]
                i_mut = individual + F * (a+b-c-d)

            # Krizeni ....
            i3 = population[np.random.randint(low=0, high=len(population))]
            for i in range(i_mut.shape[1]):
                if(np.random.uniform() < (1-CR)):
                    i_mut[0, i] = i3[0, i]

            new_population.append(i_mut)

        # Selection ...
        fits_objs = list(map(fitness_f, [x[0, :] for x in new_population]))
        evals += len(population)
        if log:
            log.add_gen(fits_objs, evals)
        new_fit = [f.fitness for f in fits_objs]  # fitness function
        for i in range(len(population)):
            if new_fit[i] >= old_fit[i]:
                population[i] = new_population[i]

    assert population[0].shape[1] == 10
    return [p[0, :] for p in population]  # only x, not sigma, alpha


def fit_gradient(pop, fitness_f, lr, get_improved=False):
    population = []

    improver = MutationGradient(fitness_f, lr)
    for ind in pop:
        population.append(improver(ind))
    fits_objs = list(map(fitness_f, [x[0, :] for x in population]))
    if not get_improved:
        return fits_objs
    else:
        return population

# def lamarck_baldwin(population, max_gen, parental_selection_algo, pop_size, crossover_wrapper, fitness_f, lr, mut_prob, mut_step_size, type):  # mutation improvement
#     if type == "lama":
#         mut_impl = MutationGradient(fitness_f,lr)
#     else:
#         mut_impl = MutationConstant(mut_step_size)
#
#     evals = 0
#     # For each generation
#     for G in range(max_gen):
#
#         # 1. Parental selection
#         # x_population = population[0, :] # first row is for the xses ..
#         if type == "lamarck":
#             fits_objs = list(map(fitness_f, [x[0, :] for x in population]))
#         elif type == "baldwin":
#             fits_objs, _ = fit_gradient(fitness_f, population, lr)
#         else:
#             raise NotImplementedError()
#
#         evals += len(population)
#         if log:
#             log.add_gen(fits_objs, evals)
#         fits = [f.fitness for f in fits_objs]  # fitness function
#         objs = [f.objective for f in fits_objs]  # objective function (we do not know this ... it is blackbox)
#
#         mating_pool_of_selected_parents = parental_selection_algo(population, fits, pop_size) # invariant to individual shape
#
#         # 2. Crossover -> result is new childs
#         offsprings = crossover_wrapper(mating_pool_of_selected_parents)  # now also invariant to individual shape
#
#         # 3. Mutation -- lamarck have some clever mutation
#         offsprings = mutation_wrapper_general(offsprings, lr, mut_prob, mut_impl)
#
#     if type == "lamarck":
#         return [p[0, :] for p in population]  # only x, not sigma, alpha
#     elif type == "baldwin":
#         _, population = fit_gradient(fitness_f, population, lr)
#         return [p[0, :] for p in population]  # only x, not sigma, alpha
#     else:
#         raise NotImplementedError()


class MutationGradient(MutationInterface):

    def __init__(self, f, lr):
        self.f = f
        self.lr = lr

    def __call__(self, ind):
        ind[0] = ind[0] - self.lr * cf.numerical_derivative(self.f,  ind[0,:])
        return ind  # ind - self.lr * cf.numerical_derivative(self.f, ind)

def mutation_wrapper_general(pop, mut_prob, mut_impl):
    offsprings = []
    for p in pop:
        if random.random() < mut_prob:
            offsprings.append(mut_impl(p))
        else:
            offsprings.append(p[:])
    assert offsprings[0].shape[1] == 10
    return offsprings


def get_fit_objs(fitness_f, special, population):
    if special != "baldwin":
        return list(map(fitness_f, [x[0, :] for x in population]))
    return fitness_f(pop=population)

def evolutionary_algorithm(population_creator, pop_size, max_gen, fitness_f, crossover_wrapper, mut_impl, mut_prob, parental_selection_algo, *, log=None, num=1, special=None, F=0.8, CR=0.9, F_decay=1, CR_decay=1, from4=False, lr=0.1, mut_step_size=0.5):
    population = population_creator.init_population(pop_size)
    if special is not None:
        if special == 'differential':
            population = differential_evo(population, max_gen, fitness_f, F=F, CR=CR, F_decay=F_decay, CR_decay=CR_decay, from4=from4, log=log)
            bi = max(population, key=fitness_f)
            return population, bi
        # if special == 'lamarck' or special == 'baldwin':
        #     return lamarck_baldwin(population, max_gen, parental_selection_algo, pop_size, crossover_wrapper, fitness_f, lr, mut_prob, mut_step_size, type=special)

    evals = 0
    POP_LLEN = len(population)
    # For each generation
    for G in range(max_gen):
        # x_population = population[0, :] # first row is for the xses ..

        fits_objs = get_fit_objs(fitness_f, special, population)
        evals += len(population)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]  # fitness function
        objs = [f.objective for f in fits_objs]  # objective function (we do not know this ... it is blackbox)

        # Parental selection
        # For ES this is random ... (no fits used ..)
        mating_pool_of_selected_parents = parental_selection_algo(population, fits, pop_size) # invariant to individual shape

        # >>>>>>> C R O S S O V E R <<<<<<<<
        # for ES, it would be good to create some grater number of offsprings
        offspringsS = crossover_wrapper(mating_pool_of_selected_parents)  # now also invariant to individual shape

        # >>>>>>>> M U T A T I O N <<<<<<<<
        all_offsprings = []
        for i in range(num):  # len(population)
            # fit before
            fitness_before_mutation = [f.fitness for f in get_fit_objs(fitness_f, special, offspringsS) ]
            # mutate
            offsprings = mutation_wrapper_general(offspringsS, mut_prob, mut_impl)

            # fit after
            fitness_after_mutation = [f.fitness for f in get_fit_objs(fitness_f, special, offsprings)]
            # update mutation params...
            mut_impl.at_episode_end(fitness_before_mutation, fitness_after_mutation)
            all_offsprings.extend(offsprings)

            # for ES we would now have here "environmental selection"
            # either M,L (only from L offsprings)
            # or M+L (from M parents, adn L offspring)
            # but now we just take all offsprings -- that is because we have L (offsprings) == M (M parents)
        fits = [f.fitness for f in get_fit_objs(fitness_f, special, all_offsprings)]
        args = np.argsort(fits)
        this_ones = args[-POP_LLEN:]

        population = [all_offsprings[i] for i in this_ones]
        assert population[0].shape[1] == 10

    if special == "baldwin":
        # pp = []
        # improver = MutationGradient(fitness_f, lr)
        # for ind in population:
        #     pp.append(improver(ind))

        population = fitness_f(pop=population, get_improved=True)
        population = [p[0, :] for p in population]  # only x, not sigma, alpha
        bi = population[0]

    else:
        population = [p[0, :] for p in population]  # only x, not sigma, alpha
        bi = max(population, key=fitness_f)

    return population, bi



def main():
    DIMENSION = 10  # dimension of the problems

    # we will run the experiment on a number of different functions
    fit_generators = [
        cf.make_f01_sphere,
        cf.make_f02_ellipsoidal,
        cf.make_f06_attractive_sector,
        cf.make_f08_rosenbrock,
        cf.make_f10_rotated_ellipsoidal
    ]
    fit_names = [
        # 'f01',
        # 'f02',
        # 'f06',
        # 'f08',
        # 'f10'
    ]

    REPEATS = 5  # number of runs of algorithm (10+)
    MAX_GEN = 250  # maximum number of generations
    POP_SIZE = 100  # population size...

    MUT_STEP_SIGMA = 0.5  # size of the mutation steps ... initial variance for Mutation .....
    MUT_PROB = 0.2  # mutation probability .. mutate 20% of the offsprings
    CX_PROB = 0.8  # crossover probability  .. crossover 80% parrents

    experiments = [

        {"name": "const::one_pt::mutProb0.2", "mutation": MutationConstant, "crossover": one_pt_cross, "num": 1, "special": None },
        {"name": "differential::F0.8::CR0.9::linDecay", "mutation": MutationConstant, "crossover": one_pt_cross, "num": 1,  "special": 'differential', "F": 0.8, "CR": 0.9, "F_decay": 0.997, "CR_decay": .997},
        #
        {"name": "lamarck::lr0.4::mutProb0.2", "mutation": MutationGradient, "crossover": one_pt_cross, "num": 1,  "special": 'lamarck', "lr":0.5, "MUT_PROB": 0.2},
        {"name": "lamarck::lr0.4::mutProb0.9", "mutation": MutationGradient, "crossover": one_pt_cross, "num": 1,
         "special": 'lamarck', "lr": 0.5, "MUT_PROB": 0.9},
        {"name": "baldwin::lr0.4::mutProb0.2", "mutation": MutationConstant, "crossover": one_pt_cross, "num": 1, "special": 'baldwin', "lr": 0.5},
        # todo - zmenit tak, ze nebudeme volat baldwin etc ...
        # ale nechame to v tom hlavnim loopu - upravit Mutation .. ( pro lamarcka)
        # upravit fitness, pro baldwina ...

        # {"name": "differential::0.8::0.9", "mutation": MutationConstant, "crossover": one_pt_cross, "num": 1, "special": 'differential', "F":0.8, "CR":0.9, "F_decay":1, "CR_decay":1},

        # {"name": "differential::0.8::0.9::from4", "mutation": MutationConstant, "crossover": one_pt_cross, "num": 1,   "special": 'differential', "F": 0.8, "CR": 0.9, "F_decay": 1, "CR_decay":1, "from4":True},

        # {"name": "scale_rotate_ind::none::M,10L", "mutation": RotatableIndividual, "crossover": no_cross,  "num": 10}, # todo - done
        # {"name": "scale_ind::none::M,10L", "mutation": MutationScalableIndividual, "crossover": no_cross, "num": 10},
        #
        # {"name": "scale_rotate_ind::none", "mutation": RotatableIndividual, "crossover": no_cross,  "num": 1},
        # {"name": "scale_ind::none", "mutation": MutationScalableIndividual, "crossover": no_cross, "num": 1},
        #
        # {"name": "lin_decay::one_pt",     "mutation": MutationLinearDecay,  "crossover": one_pt_cross, "num": 1},
        #    {"name": "rule_20::one_pt",   "mutation": MutationRule20,       "crossover": one_pt_cross, "num": 1},
        #    {"name": "scalable::one_pt", "mutation": MutationScalable, "crossover": one_pt_cross, "num": 1},
        #
        #
        #    {"name": "const::weighted", "mutation": MutationConstant, "crossover": weighted_cross, "num": 1},
        #    {"name": "lin_decay::weighted", "mutation": MutationLinearDecay, "crossover": weighted_cross, "num": 1},
        #    {"name": "rule_20::weighted", "mutation": MutationRule20, "crossover": weighted_cross, "num": 1},
        #    {"name": "scalable::weighted", "mutation": MutationScalable, "crossover": weighted_cross, "num": 1},
        #
        #    {"name": "scale_ind::one_pt::M,10L", "mutation": MutationScalableIndividual, "crossover": one_pt_cross, "num": 10},
        #  {"name": "scale_rotate_ind::one_pt::M,10L", "mutation": RotatableIndividual, "crossover": one_pt_cross,  "num":10},
        #
        # {"name": "scale_ind::one_pt", "mutation": MutationScalableIndividual, "crossover": one_pt_cross,
        #  "num": 1},
        # {"name": "scale_rotate_ind::one_pt", "mutation": RotatableIndividual, "crossover": one_pt_cross,
        #  "num": 1},

    ]
    # For each experiment ...
    for experiment in experiments:
        if experiment['special'] not in [None, 'lamarck', 'baldwin', 'differential']:
            raise NotImplementedError()

        EXP_ID = experiment["name"]

        OUT_DIR = 'lamarck_baldwin_cv06'  # output directory for logs

        if "MUT_PROB" in experiment:
            MUT_PROB = experiment["MUT_PROB"]

        # Crossover
        cross_impl = experiment['crossover']
        cross = functools.partial(crossover, cross=cross_impl, cx_prob=CX_PROB)  # todo -- cross == crossover impl.

        # Initial population
        individual_creator = PopulationCreator(ind_len=DIMENSION)

        # Fit all following functions ....
        for fit_gen, fit_name in zip(fit_generators, fit_names):
            # Fitness function
            fit = fit_gen(DIMENSION)

            # Mutation
            if experiment['special'] == "lamarck":
                mutation_impl = experiment["mutation"](fit,
                                                       experiment['lr'])  # mutation .. [  step size is the N(0,s) ]\
                print("lamarck")
            else:
                mutation_impl = experiment["mutation"]( step_size=MUT_STEP_SIGMA)  # mutation .. [  step size is the N(0,s) ]

            if experiment['special'] == "baldwin":
                fit = functools.partial(fit_gradient, fitness_f=fit, lr=experiment['lr'])
                print("baldwin")

            # Repeat it multiple times, save best solution from each last generation ...
            best_inds = []
            for run in range(REPEATS):
                # initialize the log structure
                log = utils.Log(OUT_DIR, EXP_ID + '.' + fit_name, run, write_immediately=True, print_frequency=5,
                                exper_name=experiment["name"])

                # run evolution - notice we use the pool.map as the map_fn
                F = 0.8
                CR = 0.9
                F_decay = 1
                CR_decay = 1
                lr = 0.1
                if experiment["special"] == 'differential':
                    F = experiment['F']
                    CR = experiment['CR']
                    F_decay = experiment['F_decay']
                    CR_decay = experiment['CR_decay']
                if experiment["special"] == 'lamarck' or experiment["special"] == 'baldwin':
                    lr = experiment["lr"]
                from4 = False
                if "from4" in experiment:
                    from4 = True
                pop, bi = evolutionary_algorithm(individual_creator, POP_SIZE, MAX_GEN, fit, cross, mutation_impl,
                                                 MUT_PROB, tournament_selection, log=log, num=experiment["num"],
                                                 special=experiment["special"], F=F, CR=CR, F_decay=F_decay,
                                                 CR_decay=CR_decay, from4=from4, lr=lr, mut_step_size=MUT_STEP_SIGMA)
                # remember the best individual from last generation, save it to file

                best_inds.append(bi)

                # if we used write_immediately = False, we would need to save the
                # files now
                # log.write_files()

            # After 10 runs
            # Print an overview of the best individuals from each run
            # for i, bi in enumerate(best_inds):
            #     print(f'Run {i}: objective = {fit(bi).objective}')

            # And write summary logs for the whole experiment
            utils.summarize_experiment(OUT_DIR, EXP_ID + '.' + fit_name)

if __name__ == '__main__':
    main()
