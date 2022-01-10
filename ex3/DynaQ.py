import numpy as np
import matplotlib.pylab as plt
from utils import rand_argmax
import nStepSARSA


class DynaQ:
    """
    Class representing the Dyna-Q agent
    """
    def __init__(self, env, planning_steps=0):
        self.size = env.size
        self.epsilon = 0.1
        self.gamma = 0.95
        self.step_size = 0.1
        self.Q = np.zeros(self.size + (env.num_actions, ))
        self.model = np.zeros(self.size + (env.num_actions, 3))
        self.visited_states = []
        self.actions_taken = np.frompyfunc(list, 0, 1)(np.empty(self.size, dtype=object))
        self.state = env.state
        self.planning_steps = planning_steps
        self.actions = list(range(env.num_actions))
        pass

    def update_state(self, s):
        """
        Updates the state member function of the Dyna-Q agent
        :param s: new state
        :return: None
        """
        self.state = s

    def update_q(self, s, r, next_state, a):
        """
        Function that updates the Q-value estimates of the agent.
        :param s: current state
        :param r: reward
        :param next_state: next state
        :param a: action taken
        :return: None
        """
        s_a = s + (a, )
        self.Q[s_a] += self.step_size * (r + self.gamma * self.Q[next_state + (rand_argmax(self.Q[next_state]), )] - self.Q[s_a])

    def update_model(self, a, r, next_s):
        """
        Function that updates the environment model of the agent
        :param a: action taken
        :param r: reward
        :param next_s: next state
        :return: None
        """
        self.visited_states.append(self.state)
        self.actions_taken[self.state].append(a)
        self.model[self.state + (a, )] = r, next_s[0], next_s[1]

    def get_action(self, s):
        """
        Returns an action based on an epsilon-greedy strategy
        :param s: current state
        :return: action to take
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return rand_argmax(self.Q[s])

    def plan(self):
        """
        Function that performs the planning steps of the Dyna-Q agent. If self.planning_steps is zero,
        i.e. direct RL, nothing is done
        :return:
        """
        for i in range(self.planning_steps):
            s = self.visited_states[np.random.choice(list(range(len(self.visited_states))))]
            a = np.random.choice(self.actions_taken[s])
            r, next_s_row, next_s_col = self.model[s + (a, )]
            self.update_q(s, r, (int(next_s_row), int(next_s_col)), a)


class DynaMaze:
    """
    Class representing the DynaMaze environment
    """
    def __init__(self):
        self.size = (7, 10)
        self.num_actions = 4
        self.reward = 0
        self.done = False
        self.state = (3, 0)
        self.actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.obstacles = [(1, 2), (2, 2), (4, 6), (0, 8), (1, 8), (2, 8)]
        self.goal = (0, 9)

    def step(self, action):
        """
        Function that performs an action on the environment
        :param action: action to take
        :return: (state, reward, done) tuple, where state is the new state, reward is the received reward
        for the performed transition and done indicated whether the terminal state has been reached.
        """
        r = 0
        drow = self.actions[action][0]
        dcol = self.actions[action][1]

        new_row = self.state[0] + drow
        new_col = self.state[1] + dcol

        if (new_row, new_col) in self.obstacles:
            return self.state, r, self.done

        if (new_row < 0 or new_row >= self.size[0]) or (new_col < 0 or new_col >= self.size[1]):
            return self.state, r, self.done
        else:
            self.state = (int(new_row), int(new_col))
            if self.state == self.goal:
                r = 1
                self.done = True
            return self.state, r, self.done

    def reset(self):
        """
        Resets the environment, i.e. sets the state to the start state (3, 0) and sets the done flag to False
        :return: None
        """
        self.state = (3, 0)
        self.done = False

    def get_state(self):
        """
        Get the current state
        :return: current state
        """
        return self.state


if __name__ == '__main__':
    """
    This main function performs the episode iterations. It instantiates the environment as well as the agent.
    Each episode takes as long until the agent has reached the goal state.
    Scenarios with different number of planning steps are looked at as well as one n-step algorithm (n-step Sarsa).
    For each of them, the number of time steps needed per episode are saved in order to subsequently plot the results
    and save them as png file.  
    """
    episodes = 50
    planning_steps = [0, 5, 50]
    times = np.zeros((len(planning_steps) + 1, episodes))
    for p in range(len(planning_steps)):
        n = planning_steps.pop(0)
        env = DynaMaze()
        agent = DynaQ(env, planning_steps=n)
        for e in range(episodes):
            t = 0
            done = False
            env.reset()
            while not done:
                t += 1
                a = agent.get_action(env.get_state())
                next_s, r, done = env.step(a)
                agent.update_q(agent.state, r, next_s, a)
                agent.update_model(a, r, next_s)
                agent.plan()
                agent.update_state(next_s)
            print("Finished episode {0} in {1} time steps".format(e, t))
            times[p][e] = t

    n = 25
    env = DynaMaze()
    multistep_agent = nStepSARSA.nStepSARSA(env, n=n)
    for e in range(episodes):
        env.reset()
        multistep_agent.state = env.get_state()
        times[3][e] = multistep_agent.run_episode()

    fig, ax = plt.subplots()
    ax.plot(times[0], label='0 planning steps')
    ax.plot(times[1], label='5 planning steps')
    ax.plot(times[2], label='50 planning steps')
    ax.plot(times[3], label='{0}-step SARSA'.format(n))
    ax.set_xlabel('episodes')
    lim = ax.get_xlim()
    ax.set_xlim((0, lim[1]))
    ax.set_ylim((0, 800))
    ax.set_ylabel('steps per episode')
    ax.set_title('Dyna-Q vs n-step SARSA learning DynaMaze')
    print("Average number of steps = {0}".format(np.mean(times)))

    # an extra tick indicating to which value the approaches converge (assuming n=0 and n=5
    # converge more or less to the same value as n=50 does
    yticks = list(plt.yticks()[0])
    yticks.remove(0)
    plt.yticks(yticks + [int(np.mean(times[2][-20:]))])
    plt.legend()
    plt.savefig('planning_vs_nstep.png', dpi=300)
    plt.show()
