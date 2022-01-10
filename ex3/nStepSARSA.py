import numpy as np
import matplotlib.pylab as plt
import DynaQ
from utils import rand_argmax


class nStepSARSA:
    """
    Class representing the n-step SARSA agent
    """
    def __init__(self, env, n=0):
        self.n = n
        self.step_size = 0.5
        self.epsilon = 0.05
        self.gamma = 1
        self.env = env
        sz = env.size
        a = env.num_actions
        self.Q = np.zeros(sz + (a, ))
        self.state = env.state
        self.next_action = np.inf
        self.T = np.inf
        self.t = 0
        self.R = np.zeros(self.n + 1, dtype=int)
        self.S = np.zeros((self.n + 1, 2), dtype=int)
        self.A = np.zeros(self.n + 1, dtype=int)
        self.once = True
        self.visited_states = np.zeros(env.size)

    def set_state(self, s):
        """
        Update the state
        :param s: new state
        :return: None
        """
        self.state = s

    def get_action(self, s):
        """
        Returns an action based on an epsilon-greedy strategy
        :param s: current state
        :return: action to take
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.env.num_actions)
        else:
            return rand_argmax(self.Q[s])

    def run_episode(self):
        """
        runs an episode for learning Q
        :return: number of time steps taken to finish the episode
        """
        tau = 0
        t = 0
        once = True
        self.T = np.inf

        a = self.get_action(self.state)
        self.visited_states[self.state] += 1

        self.S[t % (self.n)] = self.state[0], self.state[1]
        self.A[t % (self.n)] = a

        while tau < (self.T - 1):
            t += 1
            if t < self.T:
                next_s, r, done = self.env.step(a)
                self.visited_states[next_s] += 1
                self.S[t % (self.n)] = next_s[0], next_s[1]
                self.R[t % (self.n)] = r

                if done and once:
                    self.T = t
                    once = False
                else:
                    a = self.get_action(next_s)
                    self.A[t % (self.n)] = a
                self.state = next_s
            tau = t - self.n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + self.n, self.T) + 1):
                    G += (self.gamma ** (i - tau - 1)) * self.R[i % (self.n)]
                if (tau + self.n) < self.T:
                    idx = (tau + self.n) % (self.n)
                    S = (self.S[idx][0], self.S[idx][1])
                    G += (self.gamma ** self.n) * self.Q[S + (self.A[idx],)]
                s_a = (self.S[tau % (self.n)][0], self.S[tau % (self.n)][1],  self.A[tau % (self.n)])
                if True:
                    sum_td = 0
                    for i in range(tau, min(tau + self.n - 1, self.T) + 1):
                        S_A = self.S[i % (self.n)][0], self.S[i % (self.n)][1], self.A[i % (self.n)]
                        S_A2 = self.S[(i-1) % (self.n)][0], self.S[(i-1) % (self.n)][1], self.A[(i-1) % (self.n)]
                        sum_td += self.gamma**(i-tau-1)*(self.R[i % (self.n)] + self.gamma*self.Q[S_A2]) - self.Q[S_A]
                    self.Q[s_a] += self.step_size * sum_td
                else:
                    self.Q[s_a] += self.step_size * (G - self.Q[s_a])
        return self.T


if __name__ == '__main__':
    """
    this main function was only used for testing the algorithm, for comparing the algorithm
    to the planning algorithms, this file was imported in the DynaQ.py file 
    """

    episodes = 200
    env = DynaQ.DynaMaze()
    n_vals = [25] #, 50]
    times = np.zeros((len(n_vals), episodes))
    for i, n in enumerate(n_vals):
        agent = nStepSARSA(env, n=n)
        for e in range(episodes):
            env.reset()
            agent.state = env.get_state()
            t = agent.run_episode()
            print("Finished episode {} in {} time steps".format(e, t))
            times[i][e] = t

    plt.legend()
    fig, ax = plt.subplots()
    for i, n in enumerate(n_vals):
        ax.plot(times[i], label='{} steps'.format(n))
        print("Average times {}-step = {}".format(n, np.mean(times[i])))
    ax.set_xlabel('episodes')
    lim = ax.get_xlim()
    ax.set_xlim((0, lim[1]))
    lim = ax.get_ylim()
    if lim[1] > 800:
        ax.set_ylim((0, 800))
    else:
        ax.set_ylim((0, lim[1]))
    ax.set_ylabel('steps per episode')
    ax.set_title('n-Step SARSA learning DynaMaze')

    plt.legend()
    plt.savefig('nStepSarsa.png', dpi=300)
    plt.show()
