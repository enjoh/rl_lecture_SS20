import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import logging

"""
design decisions
#################
programming language:
python seems to be a very popular language in the field of machine learning and since the aim
is to also besides the lecture try out things like OpenAI gym or unity ML-agents, python seemed to be the logical choice.
having also already worked with python before, was another reason.
the code was developed and tested only under python3.8
the following python modules were used:
numpy - for data structures and random number generation
matplotlib - for plotting
multiprocessing - for parallalization
logging - for generating some basic logging output

code structure:
two classes were created, one for the agent and another one for the environment, i.e. the bandit. like this,
responsibilities could nicely be grouped.
a run_experiment function that can be parameterized initializes the agent and bandits and handles the execution
of the experiments. likes this, it is easy to perform various experiments with different parameters.
in order to speed up the execution of the code, a new process is started for every experiment. this is done using the
experiment_runner function, which was created to make the run_experiment still be callable without having to
 do anything with multiprocessing.

sequence of execution:
in the beginning the parameters and parameter combinations are defined ([epsilon,q_one]-sets, number of bandits,..)
with which the various experiments are started.
the run_experiment function then runs for the specified number of time steps and also performs as many runs as specified
and averages the results (average_reward, optimal_actions) over the number of runs.
as soon as all the processes have finished their experiment, the results are gathered and used for generating plots.
"""


class Agent:
    """
    agent class that keeps a list of current estimates and based on these selects certain actions
    """
    def __init__(self, k, epsilon=0., q_one=0.):
        """
        constructor
        :param k: number of bandits the agent interacts with
        :param epsilon: defines the degree of exploration
        :param q_one: initial estimate of Q
        """
        self.Q = np.ones(k) * q_one
        self.times_selected = np.zeros(k)
        self.alpha = 0.1
        self.epsilon = epsilon
        self.last_choice = 0

    def update_estimate(self, reward):
        """
        updates the estimates based on the reward received
        :param reward:
        :return:
        """
        self.times_selected[self.last_choice] += 1
        self.Q[self.last_choice] = self.Q[self.last_choice] + \
                                   (reward - self.Q[self.last_choice])/self.times_selected[self.last_choice]
        # self.Q = self.Q + self.alpha*(reward - self.Q)

    def get_action(self):
        """
        get the action based on the current estimates and with a probability epsilon select randomly
        :return: index of the action to take
        """
        if np.random.random() > self.epsilon:
            choice = np.argmax(self.Q)
        else:
            choice = np.random.choice(k)
        self.last_choice = choice
        return choice

    def get_estimates(self):
        return self.Q


class Bandit:
    def __init__(self, q, q_zero=0.):
        """
        constructor for a bandit
        :param q: actual q value
        :param q_zero: initial estimate for q
        """
        self.q = q
        self.Q = q_zero
        self.times_selected = 0
        self.alpha = 0.1

    def get_reward(self):
        """
        :return: reward
        """
        reward = np.random.normal(self.q, 1)
        return reward


def experiment_runner(proc, q, k=10, time_steps=1000, runs=1000, args=(0., 0.)):
    """ helper function for making it easier to start each experiment with a new process
    parameters:
    :param proc: the number of the current process
    :param q: the queue which handles the result transfer back to the main process
    :param k: how many bandits are created
    :param time_steps: how long the experiment is run
    :param runs: the number of runs of the experiment that are performed
    :param args: contains the (epsilon, q_one) sets used to start the experiment
    :return:
    """
    res = run_experiment(k, time_steps, runs, epsilon=args[0], q_one=args[1], prt=proc == 0)
    q.put((proc, res))


def run_experiment(k=10, time_steps=1000, runs=1000, epsilon=0., q_one=0., prt=False):
    """ main function that performs the experiments.

    :param k: defines how many bandits are created
    :param time_steps: how long the experiment is run
    :param runs: the number of runs of the experiment that are performed
    :param epsilon: sets the epsilon value
    :param q_one: sets the initial estimate of q, 0 per default
    :param prt: enables/disables console output, disabled per default
    :return:
    """
    average_reward = np.zeros(time_steps)
    optimal_actions = np.zeros(time_steps)
    q_estimates = np.zeros((k, time_steps))
    for run in range(runs):
        if prt:
            print("Run nr {0}".format(run))
        bandidos = []
        agent = Agent(k, epsilon, q_one)
        q_values = np.random.normal(0, 1, k)
        idx_max = int(np.argmax(q_values))

        [bandidos.append(Bandit(q_values[v], q_one)) for v in range(k)]

        for t in range(time_steps):
            q_estimates[:, t] += agent.get_estimates()
            choice = agent.get_action()
            r = bandidos[choice].get_reward()
            average_reward[t] += r
            agent.update_estimate(r)
            if choice == idx_max:
                optimal_actions[t] += 1

    q_estimates /= runs
    # experiment to visualize change in q estimates over time for optimistic initial values
    # if q_one != 0:
    #     f, a_stack = plt.subplots()
    #     final_timestep = 50
    #     a_stack.plot(np.arange(final_timestep), q_estimates[:, :final_timestep])# , baseline='weighted_wiggle')
    #     plt.savefig("streamgraph.png", dpi=300)

    average_reward /= runs
    optimal_actions /= runs
    optimal_actions *= 100
    return average_reward, optimal_actions


if __name__ == '__main__':
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)

    save_figure = False
    k = 10
    t_steps = 1000
    runs = 1000

    epsilon_values = [0.0, 0.01, 0.1, 0.5]
    q_initial = [0, 0, 0, 0]
    arguments = list(zip(epsilon_values, q_initial))
    num_procs = len(arguments)

    q = mp.Queue()
    jobs = []
    for i in range(num_procs):
        p = mp.Process(target=experiment_runner, args=(i, q, k, t_steps, runs, arguments[i], ))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print("Finished processing..")

    times = range(t_steps)
    f, a = plt.subplots()
    f1, a1 = plt.subplots()

    labels = []

    out = set()
    print("Gathering data..")
    while not q.empty():
        out = q.get()
        a.plot(times, out[1][0], label=f'$\epsilon$={arguments[out[0]][0]}, $Q_1$={arguments[out[0]][1]}', linewidth=1)
        a1.plot(times, out[1][1], label=f'$\epsilon$={arguments[out[0]][0]}, $Q_1$={arguments[out[0]][1]}', linewidth=1)
    print("Plotting results..")
    a.set_xlabel('Steps')
    a.set_ylabel('Average Reward')
    a.legend()
    a1.set_xlabel('Steps')
    a1.set_ylabel('Optimal Action %')
    a1.legend()

    if save_figure:
        f.savefig('averageReward.png', dpi=300)
        f1.savefig('optimalAction.png', dpi=300)
    else:
        f.show()
        f1.show()
    print("Finished!")
