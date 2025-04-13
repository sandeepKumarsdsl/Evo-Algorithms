# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:34:56 2021

@author: sande
"""
import random
import pprint

MAX_GEN = 100
POP_SIZE = 50
DIMENSION = 25

def create_individual():
    return [random.randint(0,1) for _ in range(DIMENSION)]

def create_random_population():
    return [create_individual() for _ in range(POP_SIZE)]

def onemax(ind):
    count = 0
    for i in range(len(ind)-1):
        if (ind[i]!=ind[i+1]):
           count+=1
    if(count >= (len(ind)-1)):       
       print("individual with Alternating fitness 0's and 1's",ind)    
    return sum(ind)

def select(pop, fits):
    return random.choices(pop, fits, k=POP_SIZE)

def cross(p1, p2):
    point = random.randint(0, DIMENSION-1)
    o1 = p1[:point]+p2[point:]
    o2 = p2[:point]+p1[point:]
   # print ("o1\n o2\n",o1,o2)
    return o1, o2

def mutate(ind):
    o = []
    for bit in ind:
        if random.random() < 0.05:
            o.append(1-bit)
        else:
            o.append(bit)
            
    return o

def mutation(pop):
    return[mutate(ind) for ind in pop]
    
def crossover(pop):
    offspring = []
    for p1,p2 in zip(pop[::4],pop[2::4]):
        o1, o2 = cross(p1, p2)
        offspring.append(o1)
        offspring.append(o2)
        
    return offspring

def mate(pop):
    pop1 = crossover(pop)
    return mutation(pop1)

def evolutionary_algorithm(fitness):
    pop=create_random_population()
    log=[]
    for G in range (MAX_GEN):
        fits = list(map(fitness, pop))
        log.append((sum(fits)/POP_SIZE,max(fits)))
        mating_pool = select(pop, fits)
        offspring = mate(mating_pool)
        pop = offspring[:] #+ [min(pop, key=fitness)]
        
    return pop, log

#print (create_individual())

#pprint.pprint(create_random_population())

pop = create_random_population()
fits = list(map(onemax, pop))
pprint.pprint (list(zip(pop, fits)))

print ('='*80)

result, log = evolutionary_algorithm(onemax)
res_fits = list(map(onemax,result))
pprint.pprint(list(zip(result,res_fits)))
pprint.pprint(log)

import matplotlib.pyplot as plt

averages = [l[0] for l in log]
best = [l[1] for l in log]

plt.plot(averages, label='AVG')
plt.plot(best, label='BEST')
plt.legend()
plt.show()

