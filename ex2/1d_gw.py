import numpy as np

"""
code structure:
the code was logically divided into a GridWorld class that models the environment and functions that perform the
policy evaluation and improvement.

dependencies:
numpy - for efficient data structures and generating random numbers

sequence of execution:
in the main function (bottom) first the environment is generated and an initial equi-probable policy is set up.
then, the value function for the policy is estimated using the function evaluate_policy().
afterwards, a second policy that always goes right, is improved to the optimal policy for this setting.
"""


class GridWorld:
    """ class that represents the 1-dimensional gridworld environment """
    def __init__(self, size):
        """
        constructor of the environment
        :param self:
        :param size: how many states shall the 1d gridworld have
        :return: nothing
        """
        self.actions = [-1, 1]
        self.reward_left = 10
        self.reward_right = -5
        self.rewards = [self.reward_left, 0, self.reward_right]
        self.num_states = size
        self.states = range(0, self.num_states)
        self.current_state = np.random.randint(1, size-1)
        self.done = False
        self.reward = 0
        pass

    def step(self, action):
        """
        perform one step in the environment
        :param self:
        :param action: action to be taken
        :return: a tuple consisting of the reward, the new state and whether a terminal state has been reached
        """
        if not self.done:
            assert(action in self.actions)
            self.reward = 0
            new_state = self.current_state + action
            if new_state == 0:
                self.reward = self.reward_left
                self.done = True
            elif new_state == self.num_states:
                self.reward = self.reward_right
                self.done = True
            else:
                self.reward = 0
            self.current_state = new_state
        else:
            self.reward = 0
        return self.reward, self.current_state, self.done

    def reset(self):
        """
        resets the environment, starts at a random non-terminal state and sets done-flag to False
        :param self:
        :return: nothing
        """
        self.current_state = np.random.randint(1, self.num_states-1)
        self.done = False
        pass

    def _get_state_reward(self, s):
        """
        returns the reward corresponding to a given state
        :param self:
        :param s: state to be evaluated
        :return: reward corresponding to the state
        """
        if s in self.states:
            if s == 0:
                return self.reward_left
            elif s == self.num_states-1:
                return self.reward_right
            else:
                return 0
        else:
            return 0

    def get_prob(self, ss, r, s, a):
        """
        calculate the probability that a certain transition occurs
        :param self:
        :param ss: next state
        :param r: reward
        :param s: current state
        :param a: action taken
        :return: probability of this transition occuring
        """
        if self._get_state_reward(ss) == r and s+a == ss:
            return 1.
        else:
            return 0.


def evaluate_policy(pi, env, theta=0.01):
    """
    evaluates a given policy on an environment
    :param pi: policy to be evaluated
    :param env: environment
    :param theta: tolerance parameter to check convergence
    :return:
    """
    gamma = 1
    V = np.zeros(env.num_states)
    rewards = [env.reward_left, 0, env.reward_right]
    actions = env.actions

    it = 0
    while True:
        it += 1
        delta = 0
        for s in range(1, env.num_states-1):
            v = V[s]
            V[s] = 0
            for a in actions:
                for r in rewards:
                    V[s] += pi[s, a] * env.get_prob(s+a, r, s, a)*(r + gamma*V[s+a])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            print("Policy evaluation converged after {0} iterations".format(it))
            print("Approximate value function is:\n{}".format(V))
            print("delta = {}".format(delta))
            return V


def improve_policy(policy, env, V):
    """
    improves a given policy
    :param policy: policy to be improved
    :param env: environment
    :param V: value function
    :return: stable, improved policy
    """
    gamma = 1
    it = 0
    while True:
        it += 1
        policy_stable = True
        for s in range(1, env.num_states - 1):
            old_action = policy[s]
            max_v = 0
            max_v_idx = 0
            for idx, a in enumerate(env.actions):
                value = 0
                for r in env.rewards:
                    value += env.get_prob(s + a, r, s, a) * (r + gamma * V[s + a])

                if value > max_v:
                    max_v = value
                    max_v_idx = idx

            policy[s] = env.actions[max_v_idx]
            if old_action == policy[s]:
                policy_stable = True and policy_stable
            else:
                policy_stable = False
        if policy_stable:
            print("\n\nPolicy improvement converged after {} iterations".format(it))
            print("Stable policy is {}".format(policy))
            return policy


def policy_iteration(policy, env):
    """
    performs policy iteration - repeatedly evaluates and improves a given policy
    :param policy: policy to be evaluated and improved
    :param env: environment
    :return: nothing
    """
    for i in range(10):
        evaluate_policy(policy, env)
        improve_policy()


if __name__ == '__main__':
    states = 10
    actions = 2
    env = GridWorld(states)
    policy = np.ones((states, actions)) * 0.5  # equi-probable policy

    V = evaluate_policy(policy, env)
    p = np.zeros(states) # deterministic policy
    p[:] = 1  # always go right
    p[0] = 0  # no action in terminal states
    p[-1] = 0
    new_policy = improve_policy(p, env, V)
    V = evaluate_policy(new_policy, env)


