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
        algo='UCB',
        p_mean=0.5,
    ):
        self.k = k_arm
        self.time = 0
        self.initial = initial
        self.algo = algo
        self.p_mean = p_mean

    def reset(self):
        # we'll use self.q_true to track the real reward for each action
        self.q_true = np.random.uniform(low=-1, high=1, size=self.k)

        # reset estimation for each action
        # we'll use self.q_estimation to denote the reward estimation for each action.
        self.q_estimation = np.full(self.k, self.initial)

        # number of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.time = 0

    # get an action for this bandit
    # returns a list of tuples (arm, proportion) where proportions sum to 1
    def act(self):
        UCB_arm = np.argmax(self.q_estimation + np.sqrt(4 * np.log(self.time) / self.action_count))
        best_mean_arm = np.random.choice(np.where(self.q_estimation == np.max(self.q_estimation))[0])
        if self.algo == 'UCB':
            q_best = [(UCB_arm, 1)]
        elif self.algo == 'Ours':
            if best_mean_arm == UCB_arm:
                q_best = [(UCB_arm, 1)]
            else:
                q_best = [(best_mean_arm, self.p_mean), (UCB_arm, 1 - self.p_mean)]

        return q_best

    # take an action, update estimation for this action
    def step(self, action):
        reward = 0
        for arm, p in action:
            # generate the reward under Unif(real_reward - 1, real_reward + 1)
            # reward += (np.random.uniform(low = -1, high = 1) + self.q_true[arm]) * p
            # generate the reward under N(real_reward, 1)
            reward += (np.random.normal(self.q_true[arm], 1)) * p
            self.action_count[arm] += p
        
        self.time += 1
        
        # calculate c
        numerator = reward
        denominator = 0
        if self.algo == 'Ours':
            for arm, p in action:
                numerator -= p * self.q_estimation[arm]
                denominator += p ** 2
            c = numerator/denominator

        for arm, p in action:
            # update estimation using sample averages
            if self.algo == 'UCB' or len(action) == 1:
                self.q_estimation[arm] += (reward - self.q_estimation[arm]) / self.action_count[arm]
            elif self.algo == 'Ours':
                self.q_estimation[arm] += p * p * c / self.action_count[arm]
        
        return reward


def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for j in range(bandit.k):
                action = [(j, 1)]
                reward = bandit.step(action)
                rewards[i, r, j] = reward
            for t in range(bandit.k, time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
    mean_rewards = rewards.mean(axis=1)
    return mean_rewards

def project(runs=2000, time=1000):
    bandits = [Bandit(), Bandit(algo='Ours')]
    labels = ["UCB", "Ours"]
    rewards = simulate(runs, time, bandits)
    plt.figure(figsize=(20, 8))
    for (label, reward) in zip(labels, rewards):
        plt.plot(reward, label=label)
        # print(reward)
    plt.xlabel("Steps", fontsize=20)
    plt.ylabel("Average Reward (Over 2000 Runs)", fontsize=20)
    plt.title("UCB vs. Ours", fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("UCB_vs_Ours.png")
    plt.close()

if __name__ == "__main__":
    project()