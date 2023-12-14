import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange
from math import factorial


class Bandit:
    # @k_arm: # of arms
    # @initial: initial estimation for each action
    # @algo: 'UCB' or 'FUCB'
    # @p_mean: weight assigned to best_mean_arm in FUCB
    # @half_life: alternative way of specifying p_mean = 2^(-t/half_life)
    # @reward_dist: 'Normal' for r ~ N(real_reward, 1) or 'Uniform' for r ~ Unif(real_reward - 1/4, real_reward + 1/4)
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
                # generate the reward under Unif(real_reward - 1/4, real_reward + 1/4)
                reward += (np.random.uniform(low = -1/4, high = 1/4) + self.q_true[arm]) * p
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

# generate the bandits and graph labels/title for UCB vs. FUCB with constant proportions
def constant_proportions(p_means):
    bandits = [Bandit()]
    labels = ['UCB']
    for p_mean in p_means:
        bandits.append(Bandit(algo='FUCB', p_mean=p_mean))
        labels.append('FUCB p=' + str(p_mean))
    title = 'Constant Proportions'
    return bandits, labels, title

# generate the bandits and graph labels/title for UCB vs. FUCB with exponentially decaying proportions
def decaying_proportions(half_lives):
    bandits = [Bandit()]
    labels = ['UCB']
    for half_life in half_lives:
        bandits.append(Bandit(algo='FUCB', half_life=half_life))
        labels.append('FUCB half_life=' + str(half_life))
    title = 'Decaying Proportions'
    return bandits, labels, title

# find time step of min point of dip
def find_min(reward):
    sg = savitzky_golay(savitzky_golay(reward, window_size=49, order=1), window_size=49, order=1)
    min_value = 1
    for i in range(70, len(sg)):
        if sg[i] < min_value:
            min_value = sg[i]
            min_index = i
    return min_index

# smooth
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

# generate graphs
def beautify_graph(name, cols, labels, title, type, runs, time):
    df = pd.read_csv(name + '.csv', index_col='Unnamed: 0').T
    df.loc[str(len(df))] = df.iloc[-1]
    df = df[cols]
    if len(cols) <= 4:
        c = [plt.cm.BuPu([0.8]), plt.cm.RdPu([0.6]), plt.cm.GnBu([0.8]), plt.cm.BuGn([0.8])]
    else:
        c = plt.cm.plasma(np.linspace(0, 0.8, len(cols)))
    for i in range(len(cols)):
        if type == 'Regret':
            plt.plot(df[cols[i]], linewidth=1.5, c=c[i], label=labels[i])
        elif type == 'Reward':
            plt.plot(df[cols[i]], alpha=0.2, c=c[i])
            plt.plot(savitzky_golay(df[cols[i]], 49, 1), alpha=1, linewidth=1.5, c=c[i], label=labels[i])
    plt.xticks(range(0, time + 1, time // 8))
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Average ' + type + ' (Over ' + str(runs) + ' Runs)')
    plt.title('UCB vs. FUCB with ' + title)
    plt.savefig(name + '.png')
    plt.close()

def find_intersection(regret1, regret2):
    for t in range(len(regret2)-1, -1, -1):
        if regret1[t] > regret2[t]:
            return t
    return len(regret1)

def project(runs=20, time=100):
    # change params to list of p_means or list of half_lives depending on constant_proportions or decaying_proportions
    params = [100, 200, 400]
    bandits, labels, title = decaying_proportions(params)
    rewards, regrets = simulate(runs, time, bandits)

    # save data to csv
    rewards_name = title + '_rewards_' + str(params)
    regrets_name = title + '_regrets_' + str(params)
    pd.DataFrame(rewards).to_csv(rewards_name + '.csv')
    pd.DataFrame(regrets).to_csv(regrets_name + '.csv')

    # save graphs
    beautify_graph(rewards_name,
                   list(range(len(rewards))),
                   labels,
                   title,
                   'Reward',
                   runs,
                   time)

    beautify_graph(regrets_name,
                   list(range(len(regrets))),
                   labels,
                   title,
                   'Regret',
                   runs,
                   time)
    
    if title == 'Decaying Proportions':
        dips = []
        for i in range(1, len(rewards)):
            dips.append(find_min(rewards[i]))
        plt.figure(figsize=(10, 8))
        plt.plot(params, dips)
        plt.xlabel('Half Life')
        plt.ylabel('Timestep')
        plt.title(f'Time of Minimum Reward of FUCB with {title}')
        plt.savefig(title + '_dips_' + str(params) + '.png')
        plt.close()
    else:
        intersections = []
        for i in range(1, len(regrets)):
            intersections.append(find_intersection(regrets[0], regrets[i]))
        plt.figure(figsize=(10, 8))
        plt.plot(params, intersections)
        plt.xlabel("p")
        plt.ylabel("Timestep")
        plt.title(f"Time until regret of FUCB with {title} exceeds UCB")
        plt.savefig(title + '_Regret_Intersection_' + str(params) + '.png')
        plt.close()

if __name__ == "__main__":
    project()