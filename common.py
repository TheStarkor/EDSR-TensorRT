import matplotlib.pyplot as plt

BATCH_SIZE = 1
IMAGE_SIZE = 360
ITERATION = 1000

def plot(x, y, filename='res.png', title='title', xlabel='xlabel', ylabel='ylabel', legend=['legend']):

    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legend)
    plt.savefig(filename, dpi=300)