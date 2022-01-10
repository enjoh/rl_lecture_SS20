import numpy as np
import matplotlib.pylab as plt

"""
code structure:
the code was divided into a class for the agent, MonteCarloAgent, and a class for the environment, Racetrack.
the idea is, that after instantiation of the environment, it is handed to the constructor of the agent class,
which then internally performs interactions with it.
the structure of the racetrack is defined in a racetrack.csv, which is read into a numpy array in the constructor
of the RaceTrack class.


sequence of execution:
first, the environment has to be initiated, then the agent is initated, with the environment given as constructor argument.
then, repeatedly the function run_episode() can be called to accumulate information for the estimation of the
action-value function. this information can be used to improve the current policy with the function improve_policy().

"""


class MonteCarloAgent:
    def __init__(self, env):
        """
        constructor of the agent class, that initializes all the necessary members variables
        :param env: environment, the agent shall interact with
        """
        self.env = env
        self.epsilon = 0.1
        self.episode = []
        self.gamma = 0.9
        self.policy = np.zeros((self.env.track.shape[0], self.env.track.shape[1]))
        self.Q = np.zeros((self.env.track.shape[0], self.env.track.shape[1], self.env.num_speeds, self.env.num_speeds,
                           self.env.num_actions))
        self.N = np.zeros((self.env.track.shape[0], self.env.track.shape[1], self.env.num_speeds, self.env.num_speeds,
                           self.env.num_actions))
        self.visited_positions = np.zeros((self.env.track.shape[0], self.env.track.shape[1]))

    def run_episode(self, learn=True):
        """
        runs one episode
        :param learn: parameter to distinguish between runs, where the agent shall learn and improve its
        action value function by creating episodes with a random policy, and runs, where the policy based on the
         action value function shall be used to create an episode
        :return: nothing
        """
        done = False
        state = env.reset()
        self.episode = []
        self.visited_positions *= 0
        while not done:
            if learn or np.random.uniform(0, 1) < self.epsilon:
                a = np.random.randint(0, self.env.num_actions)
            else:
                a = self.policy[state]
            s, r, done = env.step(a)
            self.episode.append((state, a, r))
            if not learn:
                self.visited_positions[state[0], state[1]] = 100
            state = s
        print("############################################")
        if learn:
            print("Finished learning episode in {} iterations.".format(len(self.episode)))
            retrn = 0
            for i in reversed(range(len(self.episode))):
                ep = self.episode[i]
                r = ep[2]
                s_a = ep[0] + (ep[1],)
                self.N[s_a] += 1
                retrn = r + self.gamma * retrn
                self.Q[s_a] += (retrn - self.Q[s_a]) / self.N[s_a]
        else:
            print("Finished non-learning episode in {} iterations.".format(len(self.episode)))
        print("############################################\n")

    def improve_policy(self):
        """
        improves the agents policy based on the previously generated action value function
        :return: nothing
        """
        self.policy = np.argmax(self.Q, axis=-1)

    def plot_max_action_values(self):
        """
        plot the current action values
        :return: nothing
        """
        img = np.max(self.Q, axis=(2, 3, 4))
        plt.imshow(img)
        plt.colorbar()
        plt.show()

    def plot_last_run(self):
        """
        plots the racetrack with the visited positions in the last episode
        :return: nothing
        """
        plt.imshow(self.visited_positions)
        plt.colorbar()
        plt.show()

    def plot_policy(self):
        for i in range(self.env.num_speeds):
            for j in range(self.env.num_speeds):
                plt.title("policies for velocity {:d} {:d}".format(i, j))
                plt.imshow(self.policy[:, :, i, j])
                plt.colorbar()
                plt.show()


class Racetrack:
    """
    environment class modeling the racetrack scenario
    """
    def __init__(self):
        """
        constructor initializing the necessary member variables.
        after reading in the racetrack from the file racetrack.csv, a plot of the track is shown.
        """
        self.track = np.genfromtxt('racetrack.csv', delimiter=',')
        self.track = np.rot90(self.track, 3)
        self.visited_positions = np.zeros(self.track.shape)
        self.off_track = np.min(self.track)
        self.finish_line = np.max(self.track)
        self.start_positions = np.where(self.track == 100)
        self.end_positions = np.where(self.track == 200)
        self.visited_positions[self.start_positions] += 1
        self.num_actions = 9
        self.num_speeds = 5
        self.posx = 0
        self.posy = 0
        self.vx = 0
        self.vy = 0
        self.actions = [-1, 0, 1]
        self.done = False
        self.reset()
        self.fig, self.ax = plt.subplots()
        self.ax.matshow(self.track)
        plt.show()

    def reset(self):
        """
        resets the environment, sets velocities to 0 and randomly selects a new start position out of
        the set of possible ones
        :return: x and y coordinates of the new start position
        """
        idx = np.random.randint(0, len(self.start_positions[0]))
        self.done = False
        self.posx = self.start_positions[0][idx]
        self.posy = self.start_positions[1][idx]
        self.vx = 0
        self.vy = 0
        # print("Start position is : column {0}, row {1}".format(self.posx, self.posy))
        return self.posx, self.posy, self.vx, self.vy

    def step(self, action):
        """
        perform one step, i.e. take the given action
        :param action: action to be taken
        :return: set consisting of the new state, the reward and a flag indicating whether a terminal state has been reached
        """
        assert(action < self.num_actions)
        reward = 0
        y, x = divmod(action, 3)
        dx = self.actions[x]
        dy = self.actions[y]
        new_vx = self.vx + dx
        new_vy = self.vy + dy
        if new_vy == 0 and new_vx == 0:
            return (self.posx, self.posy, self.vx, self.vy), reward, self.done
        if new_vx != 0 and self.num_speeds > new_vx > 0:
            self.vx += dx
        if new_vy != 0 and self.num_speeds > new_vy > 0:
            self.vy += dy

        new_posx = self.posx + self.vx
        new_posy = self.posy + self.vy
        if new_posy in self.end_positions[1] and new_posx >= max(self.end_positions[0]):
            print("Passed finish line!")
            self.posx = new_posx
            self.posy = new_posy
            self.done = True
        elif 0 > new_posx or new_posy < 0 or new_posx >= self.track.shape[0] or new_posy >= self.track.shape[1] or self.track[new_posx, new_posy] == self.off_track:
            # print("Currently at {}, {}, would go to {}, {}".format(self.posx, self.posy, new_posx, new_posy))
            # print("Looks like you got off track, back to the start!")
            self.reset()
            pass
        else:
            self.posx = new_posx
            self.posy = new_posy
            # print("New position: {}, {}, vx = {}, vy = {}".format(self.posx, self.posy, self.vx, self.vy))
            self.visited_positions[self.posx, self.posy] += 1
            reward = -1
            pass
        return (self.posx, self.posy, self.vx, self.vy), reward, self.done

    def get_random_start_position(self):
        """
        randomly select one of the possible start positions
        :return: x and y coordinate of the new start position
        """
        return np.random.randint(0, len(self.start_positions[0])), np.random.randint(0, len(self.start_positions[1]))


if __name__ == '__main__':
    env = Racetrack()
    agent = MonteCarloAgent(env)
    episodes = 10

    for i in range(episodes):
        if i % 10 == 0:
            print("Iteration nr {}".format(i))
        agent.run_episode(learn=True)
        agent.improve_policy()
        # agent.plot_action_values()

    agent.run_episode(learn=False)
    agent.plot_last_run()
    agent.plot_max_action_values()
    if False:
        # for all the start positions, run an episode with the improved policy
        for i in range(len(env.start_positions[0])):
            env.posx = env.start_positions[0][i]
            env.posy = env.start_positions[1][i]
            env.vx = 0
            env.vy = 0
            agent.run_episode(learn=False)
            agent.plot_last_run()


