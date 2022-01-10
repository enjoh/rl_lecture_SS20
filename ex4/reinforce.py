import gym
import numpy as np
import matplotlib.pylab as plt
import torch


class Reinforce():
    def __init__(self, env, alpha=0.001, gamma=1., baseline=True):
        self.step_size = alpha
        self.step_size_w = 1 * alpha
        self.gamma = gamma
        self.env = env
        self.num_actions = 2
        self.state_size = 4
        self.theta = torch.rand(self.state_size, self.num_actions).numpy()
        self.w = torch.rand(self.state_size, ).numpy()
        self.T = 0
        self.R = []
        self.A = []
        self.S = []
        if baseline:
            self.b = [i*self.gamma**(200-i) for i in range(200)][::-1]
        else:
            self.b = np.zeros(200)

    def get_action(self, s):
        """
        get an action for the given state
        :param s: state
        :return: action
        """
        return np.random.choice([0, 1], p=self.get_probabilities(np.array(s)))

    def v(self, s):
        """
        get the value function estimation
        :param s: state
        :return: estiamte for v
        """
        return np.dot(self.w, s)

    def rollout(self):
        """
        rolls out an episode
        :return: total reward a
        """
        self.R = []
        self.A = []
        self.S = []
        state = self.env.reset()
        done = False
        t = 0
        self.R.append(0)
        total_reward = 0
        while not done:
            a = self.get_action(state)
            self.S.append(state)
            self.A.append(a)
            s, r, done, i = env.step(a)
            total_reward += r
            self.R.append(r)
            state = s
            if done:
                self.T = t
                return total_reward
            t += 1

    def preference(self, s, a):
        """
        get the preference of the current state-action pair
        :param s: state
        :param a: action
        :return: preference
        """
        out = s.reshape(1, 4) @ self.theta[:, a]
        return out

    def get_probabilities(self, s):
        """
        get the actino probability distribution using the softmax function of pytorch
        :param s: state
        :return: action distribution
        """
        # sm = self.softmax(s)
        sm = torch.nn.Softmax()(torch.matmul(torch.from_numpy(s).float(), torch.from_numpy(self.theta).float()))
        return sm.numpy()

    def softmax(self, s):
        """
        calculates the softmax distribution
        :param s: state
        :return: softmax distribution
        """
        sm = []
        for i in range(2):
            h = self.preference(s, i)
            h -= np.max(h)
            sm.append(np.exp(h).T)
        sm /= np.sum(sm)
        return np.array(sm).reshape(2, )

    def softmax_grad(self, sm):
        """
        calculates the gradient of the softmax function - not used
        :param sm: softmax distribution
        :return: jacobian
        """
        jacobian_m = np.diag(sm)
        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = sm[i] * (1 - sm[i])
                else:
                    jacobian_m[i][j] = -sm[i] * sm[j]
        return jacobian_m

    def grad_sm(self, state):
        """
        own implementation of softmax gradient calculation - not used
        :param state: state
        :return: gradient
        """
        out = []
        for i in range(2):
            sm = self.softmax(state)
            S = sm.reshape(-1, 1)
            tmp = np.diagflat(S) - np.dot(S, S.T)
            out.append(tmp[i, :])
        return np.array(out).reshape(4, )

    def grad(self, s, a):
        """
        calculats the gradient
        :param s: state
        :param a: action
        :return:
        """
        res = torch.nn.Softmax()(torch.matmul(torch.from_numpy(s).float(), torch.from_numpy(self.theta).float()))
        d_sm = torch.diag(res) - res.view(-1, 1) * res
        d_log = d_sm[a]/res[a]
        grad = torch.from_numpy(s).view(-1, 1) * d_log
        return grad.numpy()

    def improve(self):
        """
        improves the current policy based on the previously run episode
        :return: None
        """
        for t in range(self.T):
            G = 0
            for k in range(t+1, self.T+1):
                G += self.gamma**(k - t - 1) * self.R[k]
            g = self.grad(self.S[t], self.A[t])
            # b = self.v(self.S[t])
            delta = G - self.b[t]
            self.theta += self.step_size * self.gamma**t * delta * g
            self.w -= self.step_size_w * delta * self.S[t]


if __name__ == '__main__':
    episodes = 1000
    runs = 5
    # np.random.seed(1)
    # torch.manual_seed(1)
    # env.render()
    alpha = 0.001
    gamma = 0.99
    baseline = False
    baselines = [False, True]
    print("STEP SIZE = {}".format(alpha))
    times = np.zeros((2, runs, episodes))
    for i, b in enumerate(baselines):
        for r in range(runs):
            print("Run #{}".format(r+1))
            env = gym.make('CartPole-v0')
            # env.seed(1)
            agent = Reinforce(env, alpha=alpha, gamma=gamma, baseline=b)
            x = []
            for e in range(episodes):
                bla = agent.rollout()
                agent.improve()
                times[i][r][e] = agent.T
                if (e % 50) == 0:
                    print("Finished episode {} in {} time steps".format(e, agent.T))
                    # print("theta = \n{}\n".format(agent.theta))
            # agent.step_size *= 0.95
    fig, ax = plt.subplots()

    bl = 'with'
    for i, b in enumerate(baselines):
        bl = ''
        if b:
            bl = 'with'
        else:
            bl = 'without'
        mean = times[i].mean(axis=0)
        std = times[i].std(axis=0)
        ax.plot(mean, label='{} baseline'.format(bl))
    ax.set_ylim((0, 200))
    lim = ax.get_xlim()
    ax.set_xlim((0, lim[1]))
    ax.set_ylabel('steps per episode')
    ax.set_xlabel('episodes')

    ax.set_title('REINFORCE learning CartPole-v0\n$\\alpha = {1}, \\gamma = {2}$'.format(bl, alpha, gamma))

    plt.legend()
    plt.savefig('REINFORCE_cartpole.png'.format(bl), dpi=300)
    plt.show()
