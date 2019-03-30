import matplotlib.pyplot as plt

def plot_loss(X):
    plt.plot(X)
    plt.axis([0, 1000, 0, 1])
    plt.show()