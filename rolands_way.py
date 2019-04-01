import numpy as np
import math
import random
from matplotlib import pyplot as plt

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


input_default = [[0, 0], [0, 1], [1, 0], [1, 1]]

def generate_inputs(input_size):
    normal_x = []
    offset = []

    for x in range(int(input_size/4)):
        normal_x.extend(input_default)

    for x in range(input_size):
        offset.append([-1])

    #   Generates he correct results for said inputs
    Y = np.remainder(np.sum(normal_x, axis=1), 2)
    #   Generates noise, the scale is standard deviation, so for variance of 1/4 a scale if 1/2 is required, I think.
    noise = np.random.normal(loc=0, scale=0.5, size=(input_size, 2))
    #   Adding noise
    X = normal_x + noise
    #   Adding offset as 3rd input
    X = np.append(X, offset, axis=1)
    X = np.transpose(X)

    return X, Y


#   Roland's training data, useful for checking if the algorithm produces similar results and is thus correctly implemented
# X=[[-0.310691, -0.309003, 1.25774, 1.31959, -0.0897083, -0.457115,
# 1.42524, 1.43962, -0.21377, -0.16744, 0.579612, 1.90558, 0.442017,
# 0.204012, 1.75664, 0.584128],
# [0.0164278, 0.898471, -0.231735, 0.82952, -1.02045, 1.84369, 0.111823,
# 0.28365, 0.0759174, 0.985518, 0.584378, 0.434351, 0.35245, -0.0194183,
# -0.336488, 1.45608], [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
# -1, -1, -1, -1]]



#   Roland's method from mathmatica

def predict(weight_1, weight_2, X):
    offset_vector = np.full((1, len(X[0])), -1)
    hid = logistic_function( np.dot( weight_1, X ) )
    out = logistic_function( np.dot( weight_2, np.concatenate( (hid, offset_vector) ) ) )
    return out

def neural_net(hidden_layers, X, Y):
    weight_1 = intoh(hidden_layers)
    weight_2 = htout(hidden_layers)

    error_table = np.zeros(1000)
    offset_vector = np.full((1, len(X[0])), -1)

    #   Implementation of Roland's mathmatica neural net
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

    #   Plotting line graph
    global plot_count
    global input_count
    plt.figure(input_count)
    plt.subplot(1, 3, plot_count)
    plt.plot(error_table)
    plt.axis([0, 1000, 0, 1])
    plt.title('Hidden Layer Size: ' + str(hidden_layers) + '   Input Size: ' + str(len(X[0])))

    #   Plotting output mapping
    X_1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    X_1 = np.arange(0, 1.05, 0.05)
    output_map = np.zeros((len(X_1), len(X_1)))
    for i in range(len(X_1)):
        for j in range(len(X_1)):
            output_map[i, j] = predict(weight_1, weight_2, [[X_1[i]], [X_1[j]], [-1]])
    plt.figure(input_count + 1)
    plt.subplot(1, 3, plot_count)
    plt.imshow(output_map, cmap='gray', origin='lower')
    plt.title('Hidden Layer Size: ' + str(hidden_layers) + '   Input Size: ' + str(len(X[0])))

    return weight_1, weight_2


def net_error(weight_1, weight_2, X, Y):
    out = predict(weight_1, weight_2, X)
    error = Y - out
    mse = np.mean(error*error)
    return mse
    
#   Generating testing data for the neural nets
X_test, Y_test = generate_inputs(64)

plot_count = 1
input_count = 1
same_inputs = True
for input_size in [16, 32, 64]:
    plot_count = 1
    if same_inputs == True:
        X, Y = generate_inputs(input_size)
    for hidden_layers in [2, 4, 8]:
        if same_inputs == False:
            X, Y = generate_inputs(input_size)

        weight_1, weight_2 = neural_net(hidden_layers, X, Y)
        mse = net_error(weight_1, weight_2, X_test, Y_test)
        print ('test set error: ' + str(mse))
        plot_count += 1
    input_count += 2
plt.show()


