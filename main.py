import numpy as np

input_size = 16
hidden_layers = 2

normal_x = np.random.randint(2, size=(input_size, 2))
Y = np.remainder(np.sum(normal_x, axis=1), 2)
noise = np.random.normal(scale=0.5, size=(input_size, 2))

X = normal_x + noise

print (normal_x)
print (noise)