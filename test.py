import Chromosome
import file_handler as fh
import random
import numpy as np

data = fh.read_from_file('Dataset/Dataset1.csv')

print(data[2][0])
# print(int(data[2][0]))
print(data[2][0]*2)