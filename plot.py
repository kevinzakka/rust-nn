import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X = np.load("data/X.npy").reshape(300, 2)
    y = np.load("data/y.npy").reshape(300)
    W1 = np.load("data/W1.npy").reshape(2, 100)
    W2 = np.load("data/W2.npy").reshape(100, 3)
    b1 = np.load("data/b1.npy").reshape(1, 100)
    b2 = np.load("data/b2.npy").reshape(1, 3)

    xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), sparse=False)
    space = np.vstack((xx.flatten(), yy.flatten())).T
    pred = np.dot(np.maximum(0, np.dot(space, W1) + b1), W2) + b2
    pred = np.argmax(pred, axis=1)

    plt.figure(figsize=(10, 10))
    plt.scatter(space[:, 1], space[:, 0], c=pred, zorder=-1, alpha=0.8)
    plt.scatter(X[:, 1], X[:, 0], c=y, zorder=1, edgecolors= "black")
    plt.axis('off')
    plt.savefig("decision_boundary.png", format="png", dpi=150, bbox_inches='tight')
    plt.close()

