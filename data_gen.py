import numpy as np

from neural_net import MLP


input_size = 32
hidden_layers = 8
input_default = [[0, 0], [0, 1], [1, 0], [1, 1]]
normal_x = []
offset = []

for x in range(int(input_size/4)):
    normal_x.extend(input_default)

for x in range(input_size):
    offset.append([-1])


#   Generates the random inputs and the correct results for said inputs
# normal_x = np.random.randint(2, size=(input_size, 2))
Y = np.remainder(np.sum(normal_x, axis=1), 2)
#   Generates noise, the scale is standard deviation, so for variance of 1/4 a scale if 1/2 is required, I think.
noise = np.random.normal(loc=0, scale=0.5, size=(input_size, 2))
#   Adding noise
X = normal_x + noise
#   Adding offset as 3rd input
X = np.append(X, offset, axis=1)

print (normal_x)
print (X)

mlp = MLP(hidden_layers)
mlp.fit(X, Y)