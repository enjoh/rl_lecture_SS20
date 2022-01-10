import matplotlib.pylab as plt
import numpy as np


def f(s, w):
    return np.dot(s, w)


def grad_f(s, w):
    return s


if __name__ == '__main__':

    alphas = []
    samples = [[4, 7, 1],
               [10, 6, 0],
               [20, 1, 15],
               [4, 19, 3]]
    concentrations = np.array(samples)
    credits = [3, -15, 5, 21]
    # gradient descent
    mse = []
    alpha = 0.01
    for idx in range(4):
        alphas.append(alpha)
        mse.append(0)
        w = np.zeros(concentrations.shape[1])
        # w = [0.10328828, 0.3508688,  0.18581121]
        print("alpha = {}".format(alpha))
        for i in range(concentrations.shape[0]):
            w = w + alpha * (credits[i] - f(concentrations[i], w)) * grad_f(concentrations[i], w)

        print("final weight vector = {}".format(w))
        estimation = np.zeros(len(credits))

        for i in range(concentrations.shape[0]):
            mse[idx] += (credits[i] - f(concentrations[i], w))**2
            # print("error = {}".format((credits[i] - f(concentrations[i], w))))

        alpha *= 0.8
    mse = [num/concentrations.shape[0] for num in mse]
    print("MSE = {}".format(mse))
    fig, ax = plt.subplots()
    ax.plot(alphas, mse)
    # ax.set_xticks(alphas)
    ax.set_ylabel('MSE')
    ax.set_xlabel('$\\alpha$')
    plt.show()
