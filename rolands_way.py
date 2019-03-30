import numpy as np
import math
import random

#   ---- ROLAND'S FUNCTIONS ----

#   Return 2 x input_size matrix
def logistic_function(x):
    table = np.zeros(x.shape)
    if len(x.shape) > 1:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                table[i, j] = 1 / (1 + math.exp(-x[i, j]))
    else:
        for i in range(x.shape[0]):
            table[i] = 1 / (1 + math.exp(-x[i]))
    return table

def intoh(n):
    table = np.zeros((n, 3))
    for i in range(n):
        for j in range(3):
            table[i, j] = (4 * random.uniform(0, 1)) - 2
    return table

#   Has extra for offset
def htout(n):
    table = np.zeros(n+1)
    for i in range(n+1):
        table[i] = (4 * random.uniform(0, 1)) - 2
    return table



input_size = 16
hidden_layers = 8
input_default = [[0, 0], [0, 1], [1, 0], [1, 1]]
normal_x = []
offset = []

for x in range(int(input_size/4)):
    normal_x.extend(input_default)

for x in range(input_size):
    offset.append([-1])

#   Generates the random inputs and the correct results for said inputs
Y = np.remainder(np.sum(normal_x, axis=1), 2)
#   Generates noise, the scale is standard deviation, so for variance of 1/4 a scale if 1/2 is required, I think.
noise = np.random.normal(loc=0, scale=0.5, size=(input_size, 2))
#   Adding noise
X = normal_x + noise
#   Adding offset as 3rd input
X = np.append(X, offset, axis=1)

X = np.transpose(X)

#   Roland's method from mathmatica

weight_1 = intoh(2)
weight_2 = htout(2)

error_table = np.zeros(1000)
offset_vector = np.full((1, 16), -1)


for loop in range(1000):
    hid = logistic_function( np.dot( weight_1, X ) )
    out = logistic_function( np.dot( weight_2, np.concatenate( (hid, offset_vector) ) ) )
    error = Y - out
    error_table[loop] = np.mean(error*error)
    delta_out = error * out * (1-out)
    error_hid = np.outer(weight_2, delta_out)[0:1, :]
    delta_hid = error_hid * hid * (1-hid)
    weight_2 += np.dot( delta_out, np.transpose( np.concatenate( (hid, offset_vector) ) ) )
    weight_1 += np.dot( delta_hid, np.transpose( X ) )
    if loop % 100 == 0:
        print (error_table[loop])

print (error_table)


