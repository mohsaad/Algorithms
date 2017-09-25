from __future__ import print_function, division

import util
import numpy as np
import pytorch
import matplotlib.pyplot as plt

class AutoencoderTorch(nn.module):

    def __init__(self, D, M):
        super(AutoencoderTorch, self).__init__()

        self.fc1 = nn.Linear(D, M)
        self.fc2 = nn.linear(M, D)

    def predict(self, X):
        pass


def main():
    X, Y = util.get_mnist()

    model = AutoencoderTorch(784, 300)
    model.fit(X)


    for i in range(0, 5):
        i = np.random.choice(len(X))
        x = X[i]

        im = model.predict([x]).reshape(28, 28)
        plt.subplot(1,2,1)
        plt.imshow(x.reshape(28, 28), cmap = 'gray')
        plt.title("Original")
        plt.subplot(1,2,2)
        plt.imshow(im, cmap = 'gray')
        plt.title("reconstruction")
        plt.show()

if __name__ == '__main__':
    main()
