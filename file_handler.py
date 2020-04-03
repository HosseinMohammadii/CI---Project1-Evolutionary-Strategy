import itertools
import csv
import numpy as np


def read_from_file(fn):
    file = open(fn, 'r')
    read = csv.reader(file, delimiter=',')
    data = [row[0:] for row in read]
    data.remove(data[0])
    file.close()
    # dataa = np.zeros((len(data), 2))
    dataa = []
    for i in range(len(data)):
        top = []
        for j in range(2):
            top.append(float(data[i][j]))
        dataa.append(top)
    return dataa
