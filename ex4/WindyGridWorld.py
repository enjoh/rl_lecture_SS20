import numpy as np
import matplotlib.pylab as plt
from utils import rand_argmax


class WindyGridWorld:
    """
    Class representing the Windy Gridworld environment
    """
    def __init__(self, size=(7, 10), stochastic_wind=False, do_nothing_allowed=False):
        self.size = size
        self.grid = np.zeros(self.size)
        self.wind = np.zeros(self.size)
        self.wind[:, 3:6] = 1
        self.wind[:, 6:8] = 2
        self.wind[:, 8] = 1
        self.start_state = (3, 0)
        self.state = self.start_state
        self.goal = (3, 8)
        if do_nothing_allowed:
            self.num_actions = 9
        else:
            self.num_actions = 8
        self.actions = [-1, 0, 1]
        self.done = False
        self.stochastic_wind = stochastic_wind
        pass

    def step(self, action):
        """
        Function that performs an action on the environment
        :param action: action to take
        :return: (state, reward, done) tuple, where state is the new state, reward is the received reward
        for the performed transition and done indicated whether the terminal state has been reached.
        """
        assert(action < self.num_actions)
        r = -1
        row, col = divmod(action, 3)
        drow = self.actions[row]
        dcol = self.actions[col]
        if row == 0 and col == 0:
            return self.state, r, self.done
        w = 0
        if self.stochastic_wind and self.wind[self.state] > 0:
            choices = [-1, 0, 1]
            c = np.random.choice(choices)
            w = self.wind[self.state] + c
        else:
            w = self.wind[self.state]
        new_row = self.state[0] + drow + w
        new_col = self.state[1] + dcol
        if (new_row < 0 or new_row >= self.size[0]) and (0 <= new_col < self.size[1]):
            self.state = (self.state[0], int(new_col))
            return self.state, r, self.done
        elif (0 <= new_row < self.size[0]) and (new_col < 0 or new_col >= self.size[1]):
            self.state = (int(new_row), self.state[1])
            return self.state, r, self.done
        elif (new_row < 0 or new_row >= self.size[0]) and (new_col < 0 or new_col >= self.size[1]):
            return self.state, r, self.done
        else:
            self.state = (int(new_row), int(new_col))
            # print("New state: {}".format(self.state))
            if self.state == self.goal:
                r = 0
                self.done = True
        return self.state, r, self.done

    def reset(self, stochastic_wind=False):
        """
        Resets the environment, i.e. sets the state to the start state (3, 0) and sets the done flag to False
        :return: current state
        """
        self.state = self.start_state
        self.done = False
        self.stochastic_wind = stochastic_wind
        return self.state

    def get_size(self):
        """
        Get the size of the environment
        :return: environment size
        """
        return self.size

    def get_num_actions(self):
        """
        Get the number of available actions
        :return: number of actions
        """
        return self.num_actions


class SARSA:
    """
    Class representing the SARSA agent
    """
    def __init__(self, env, alpha=0.5):
        self.step_size = alpha
        self.epsilon = 0.1
        self.gamma = 1
        self.env = env
        sz = env.get_size()
        a = env.get_num_actions()
        self.Q = np.zeros(sz + (a, ))
        self.state = env.state
        self.next_action = np.inf

    def set_alpha(self, a):
        self.step_size = a

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

    def update_estimate(self, action, r, next_s):
        """
        Function that updates the Q-value estimates of the agent.
        :param r: reward
        :param next_state: next state
        :param a: action taken
        :return: None
        """
        s_a = self.state + (action, )
        next_a = self.get_action(next_s)
        next_s_a = next_s + (next_a, )
        self.Q[s_a] = self.Q[s_a] + self.step_size * (r + self.gamma * self.Q[next_s_a] - self.Q[s_a])
        return next_a


if __name__ == '__main__':
    """
    This main function performs the episode iterations. It instantiates the environment as well as the agent.
    Each episode takes as long until the agent has reached the goal state.
    Three different scenarios are looked at concerning the number of planning steps. For each of them,
    the number of time steps needed per episode are saved in order to subsequently plot the results and save
    them as png file.  
    """
    episodes = 1000
    do_nothing = [False, True]
    alphas = [0.5**i for i in range(3)]
    alphas = [1.5, 0.5] # , 0.0005] # alphas
    times = np.zeros((len(alphas), episodes))
    stochastic_wind = True
    for i, v in enumerate(alphas):
        env = WindyGridWorld(do_nothing_allowed=True)
        state = env.reset()
        agent = SARSA(env, alpha=v)
        for e in range(episodes):
            state = env.reset(stochastic_wind=stochastic_wind)
            agent.set_alpha(v)
            agent.set_state(state)
            t = 0
            done = False
            a = agent.get_action(state)
            while not done:
                t += 1
                next_state, reward, done = env.step(a)
                next_a = agent.update_estimate(a, reward, next_state)
                state = next_state
                agent.set_state(next_state)
                a = next_a
                if agent.step_size >= 1e-6:
                    pass
                    # agent.step_size *= 0.99
            print("Finished episode {0} in {1} time steps".format(e, t))
            times[i][e] = t

    fig, ax = plt.subplots()
    for i, v in enumerate(alphas):
        ax.plot(times[i], label='$\\alpha$ = {}'.format(v))
        # mean = np.mean(times[i][-100:])
        # m = np.ones(len(times[i])) * mean
        # ax.plot(m, label='avg(no movement: {}) = {}'.format(v, mean))
    # print("Average number of steps over last 50 episodes = {0}".format(np.mean(times[-50:])))

    ax.set_ylabel('steps per episode')
    ax.set_xlabel('episodes')

    lim = ax.get_xlim()
    ax.set_xlim((0, lim[1]))
    lim = ax.get_ylim()
    ax.set_ylim((0, 1000))
    yticks = list(plt.yticks()[0])
    yticks.remove(0)
    plt.yticks(yticks + [np.mean(times)])
    plt.legend()
    if stochastic_wind:
        ax.set_title('SARSA learning WindyGridWorld w\ stochastic wind')
        plt.savefig('WindyGridWorld_stochastic_wind', dpi=300)
    else:
        ax.set_title('SARSA learning WindyGridWorld w\o stochastic wind')
        plt.savefig('WindyGridWorld_non_stochastic_wind', dpi=300)
    plt.show()
