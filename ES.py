import Chromosome
import file_handler as fh
import random
import numpy as np
import matplotlib.pyplot as plt


data = fh.read_from_file('Dataset/Dataset1.csv')

Mu = 10
# Todo change Mu's coefficient below to attain the best result
Lambda = 7*Mu
crossover_probability = 0.4
population_size = Mu
chromosome_length = 2
iteration_size = 100
lr = 0.001
minFit = []
maxFit = []
avgFit = []



def generate_initial_population():
    list_of_chromosomes = []
    for i in range(0, population_size):
        ch = Chromosome.Chromosome(chromosome_length, -10, 10)
        list_of_chromosomes.append(ch)
    return list_of_chromosomes


def generate_new_seed(populationn):
    new_parent = random.choices(populationn, weights=None, k = 2)
    return new_parent


def crossover(chromosome1, chromosome2):
    rr = random.uniform(0, 1)
    ch = Chromosome.Chromosome(chromosome_length, -10, 10)
    ch.gene[0] = chromosome1.gene[0] * rr + (1 - rr) * chromosome2.gene[0]
    ch.gene[1] = chromosome1.gene[1] * rr + (1 - rr) * chromosome2.gene[1]
    return ch


def mutation(chromosomes, iteration_s, cur_iteration, maxSigma, minSigma):

    s = maxSigma + ((minSigma - maxSigma) / iteration_s) * cur_iteration
    mut = []
    for ch in chromosomes:
        temp = s * np.random.normal(0, 1)
        ch.gene[0] += temp
        ch.gene[1] += temp
        mut.append(ch)

    return mut


def evaluate_new_generation(chromosomes):
    for ch in chromosomes:
        ch.evaluate(data)
    return chromosomes


def evaluate_new_generation2(chromosomes):
    maxx = -1000000
    minn = 10000000
    avgg = 0
    for ch in chromosomes:
        ch.evaluate(data)

        avgg += ch.score
        if ch.score > maxx:
            maxx = ch.score
        elif ch.score < minn:
            minn = ch.score
    avgg = avgg/len(chromosomes)

    maxFit.append(maxx)
    minFit.append(minn)
    avgFit.append(avgg)

    return chromosomes


def choose_new_generation(populationn):
    q_tournament = 2
    popp = []
    for i in range(Mu):
        rand_survivors = random.choices(populationn, weights=None, k=q_tournament)
        rand_survivors_score = []
        for b in range(q_tournament):
            rand_survivors_score.append(rand_survivors[b].score)
        armx = np.argmax(rand_survivors_score)
        choosen = rand_survivors[armx]

        popp.append(choosen)
        populationn.remove(choosen)

    return popp


children = []
pop = []

if __name__ == '__main__':

    population = generate_initial_population()

    for n in range(0, iteration_size):
        evaluate_new_generation2(population)
        cur = 0
        children = []
        while cur < Lambda:
            parents = generate_new_seed(population)

            r = random.uniform(0, 1)
            if r < 0.4:
                new_child = crossover(parents[0], parents[1])
                children.append(new_child)
                cur += 1

        population = mutation(population, iteration_size, n, 2, 0.01)
        pop = population + children
        pop = evaluate_new_generation(pop)
        population = choose_new_generation(pop)

    x = np.linspace(0, iteration_size, iteration_size)

    plt.plot(maxFit, label='Best')
    plt.plot(avgFit, label='Average')
    plt.plot(minFit, label='Worst')
    plt.plot(minFit)

    plt.xlabel('Generation')
    plt.ylabel('fitness')

    plt.title("Nemudar")
    plt.legend()
    plt.show()
    maxs = -1000
    index = 0
    for i in range(len(population)):
        if population[i].score > maxs:
            maxs = population[i].score
            index = i

    a = population[index].gene[0]
    b = population[index].gene[1]
    print(a, b)

    c = b/a
    lis = [0, 50]
    liss = [c*k for k in lis]
    plt.plot(lis, liss,)
    datata = np.array(data).T
    lisss = [c*k for k in datata[0]]
    plt.plot(datata[0], datata[1],  '+', color='#777777')
    plt.scatter(datata[0], lisss, s=10, facecolors='none', edgecolors='r')
    plt.show()

    # for ch in population:
    #     print(ch.gene)