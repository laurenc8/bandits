import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


class Bandit:
    # @k_arm: # of arms
    # @initial: initial estimation for each action
    def __init__(
        self,
        k_arm=10,
        initial=0.0,
    ):
        self.k = k_arm
        self.time = 0
        self.initial = initial

    def reset(self):
        # we'll use self.q_true to track the real reward for each action
        self.q_true = np.random.uniform(low=-1, high=1, size=self.k)

        # reset estimation for each action
        # we'll use self.q_estimation to denote the reward estimation for each action.
        self.q_estimation = np.full(self.k, self.initial)

        # number of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.time = 0

    # get an action for this bandit using UCB
    def act(self):
        q_best = np.argmax(self.q_estimation + np.sqrt(4 / 3 * np.log(self.time) / self.action_count))

        return q_best

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.uniform(low = -1, high = 1) + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1

        # update estimation using sample averages
        self.q_estimation[action] += (
            reward - self.q_estimation[action]
        ) / self.action_count[action]
        
        return reward


def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for j in range(bandit.k):
                action = j
                reward = bandit.step(action)
                rewards[i, r, j] = reward
            for t in range(bandit.k, time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
    mean_rewards = rewards.mean(axis=1)
    return mean_rewards

def project(runs=2000, time=1000):
    bandits = [Bandit()]
    rewards = simulate(runs, time, bandits)
    plt.figure(figsize=(20, 8))
    for reward in rewards:
        plt.plot(reward)
        # print(reward)
    plt.xlabel("steps", fontsize=20)
    plt.ylabel("average reward (over 2000 runs)", fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("project.png")
    plt.close()

if __name__ == "__main__":
    project()