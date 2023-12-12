import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange


class Bandit:
    # @k_arm: # of arms
    # @initial: initial estimation for each action
    # @algo: 'UCB' or 'FUCB'
    # @p_mean: weight assigned to best_mean_arm in FUCB
    # @half_life: alternative way of specifying p_mean = 2^(-t/half_life)
    # @reward_dist: 'Normal' for r ~ N(real_reward, 1) or 'Uniform' for r ~ Unif(real_reward - 1, real_reward + 1)
    def __init__(
        self,
        k_arm=10,
        initial=0.0,
        algo='UCB',
        p_mean=0.5,
        half_life=0,
        reward_dist='Normal',
    ):
        self.k = k_arm
        self.time = 0
        self.initial = initial
        self.algo = algo
        self.p_mean = p_mean
        self.half_life = half_life
        self.reward_dist = reward_dist

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
        if self.half_life > 0:
            self.p_mean = 2 ** (-self.time / self.half_life)
        UCB_arm = np.argmax(self.q_estimation + np.sqrt(8 * np.log(self.time) / self.action_count))
        best_mean_arm = np.random.choice(np.where(self.q_estimation == np.max(self.q_estimation))[0])
        if self.algo == 'UCB':
            q_best = [(UCB_arm, 1)]
        elif self.algo == 'FUCB':
            if best_mean_arm == UCB_arm:
                q_best = [(UCB_arm, 1)]
            else:
                q_best = [(best_mean_arm, self.p_mean), (UCB_arm, 1 - self.p_mean)]

        return q_best

    # take an action, update estimation for this action
    def step(self, action):
        reward = 0
        for arm, p in action:
            if self.reward_dist == 'Normal':
                # generate the reward under N(real_reward, 1)
                reward += (np.random.normal(self.q_true[arm], 1)) * p
            elif self.reward_dist == 'Uniform':
                # generate the reward under Unif(real_reward - 1, real_reward + 1)
                reward += (np.random.uniform(low = -1, high = 1) + self.q_true[arm]) * p
            self.action_count[arm] += 1
        
        self.time += 1
        
        # calculate c
        numerator = reward
        denominator = 0
        if self.algo == 'FUCB' and self.time > self.k:
            for arm, p in action:
                numerator -= p * self.q_estimation[arm]
                sigma_sq = 2 * np.log(self.time) / (self.action_count[arm] - 1)
                denominator += p ** 2 * sigma_sq
            c = numerator/denominator

        for arm, p in action:
            # update estimation using sample averages
            if self.algo == 'UCB' or len(action) == 1:
                self.q_estimation[arm] += (reward - self.q_estimation[arm]) / self.action_count[arm]
            elif self.algo == 'FUCB':
                sigma_sq = 2 * np.log(self.time) / (self.action_count[arm] - 1)
                self.q_estimation[arm] += p * c * sigma_sq / self.action_count[arm]
        
        return reward


def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    regrets = np.zeros((len(bandits), runs, time))
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for j in range(bandit.k):
                action = [(j, 1)]
                reward = bandit.step(action)
                rewards[i, r, j] = reward
                regrets[i, r, j] = regrets[i, r, j - 1] + max(bandit.q_true) - reward if j > 0 else max(bandit.q_true) - reward
            for t in range(bandit.k, time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                regrets[i, r, t] = regrets[i, r, t - 1] + max(bandit.q_true) - reward
    mean_rewards = rewards.mean(axis=1)
    mean_regrets = regrets.mean(axis=1)
    return mean_rewards, mean_regrets

def constant_proportions():
    bandits = [Bandit(), Bandit(algo='FUCB', p_mean=0.1), Bandit(algo='FUCB', p_mean=0.5), Bandit(algo='FUCB', p_mean=0.9)]
    labels = ["UCB", "FUCB p=0.1", "FUCB p=0.5", "FUCB p=0.9"]
    title = 'Constant Proportions'
    return bandits, labels, title

def decaying_proportions():
    bandits = [Bandit(), Bandit(algo='FUCB', half_life=100), Bandit(algo='FUCB', half_life=200), Bandit(algo='FUCB', half_life=400)]
    labels = ["UCB", "FUCB half_life=100", "FUCB half_life=200", "FUCB half_life=400"]
    title = 'Decaying Proportions'
    return bandits, labels, title

def project(runs=4000, time=2000):
    bandits, labels, title = decaying_proportions()
    rewards, regrets = simulate(runs, time, bandits)
    pd.DataFrame(rewards).to_csv(title + '_rewards.csv')
    pd.DataFrame(regrets).to_csv(title + '_regrets.csv')

    plt.figure(figsize=(10, 8))
    for (label, reward) in zip(labels, rewards):
        plt.plot(reward, label=label)
    plt.xlabel("Steps", fontsize=20)
    plt.ylabel("Average Reward (Over " + str(runs) + " Runs)", fontsize=20)
    plt.title("UCB vs. FUCB with " + title, fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("UCB_vs_FUCB_Reward.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    for (label, regret) in zip(labels, regrets):
        plt.plot(regret, label=label)
    plt.xlabel("Steps", fontsize=20)
    plt.ylabel("Average Regret (Over " + str(runs) + " Runs)", fontsize=20)
    plt.title("UCB vs. FUCB with " + title, fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig("UCB_vs_FUCB_Regret.png")
    plt.close()

if __name__ == "__main__":
    project()