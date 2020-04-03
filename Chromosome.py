import random
import numpy as np


class Chromosome:
    def __init__(self, chromosome_length, min, max):
        # Todo create a random list for genes between min and max below
        self.gene = [random.uniform(min, max) for i in range(0, chromosome_length)]
        self.score = 0
        self.sigma = 1

    def evaluate(self, data):
        zz = []

        a = self.gene[0]
        b = self.gene[1]
        c = np.sqrt(a**2 + b**2)
        a = a / c
        b = b / c
        for dt in data:
            # print(dt[0])
            # print(type(dt[0]))
            # print(self.gene[0])
            # print(type(self.gene[0]))

            z = a * dt[0] + b * dt[1]
            zz.append(z)

        self.score = np.std(zz)

    # def evaluate(self, dInput):
    #     """
    #     Update Score Field Here
    #     """
    #     #Todo
    #     z = np.empty(dInput[:,0].size)
    #     a = self.gene[0]
    #     b = self.gene[1]
    #
    #     normalA = a/np.sqrt(a**2+b**2)
    #     normalB = b/np.sqrt(a**2+b**2)
    #
    #     for d in range(0,len(dInput)):
    #       z[d] = normalA * dInput[d, 0] + normalB * dInput[d, 1]
    #
    #     self.score = np.std(z)
