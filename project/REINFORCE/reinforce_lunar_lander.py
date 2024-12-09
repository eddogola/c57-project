import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)


class REINFORCE:
    def __init__(self, env_name='LunarLander-v2', hidden_dim=128, learning_rate=0.001, gamma=0.99, window_size=100, gradient_threshold=0.005):
        self.env = gym.make(env_name)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n
        self.gamma = gamma

        self.policy = PolicyNetwork(self.input_dim, hidden_dim, self.output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.episode_rewards = []
        self.rolling_avg_rewards = []
        self.window_size = window_size
        self.gradient_threshold = gradient_threshold
        self.convergence_episode = None

    def calculate_gradient(self):
        """Calculate the gradient of the rolling average."""
        if len(self.rolling_avg_rewards) < self.window_size:
            return None
        x = np.arange(self.window_size)
        y = np.array(self.rolling_avg_rewards[-self.window_size:])
        gradient = np.polyfit(x, y, 1)[0]
        return gradient

    def check_convergence(self):
        """Check if the gradient of the rolling average is below the threshold."""
        if len(self.rolling_avg_rewards) < max(self.window_size, 200):
            return False

        gradient = self.calculate_gradient()
        if gradient is not None and abs(gradient) < self.gradient_threshold:
            return True
        return False

    def select_action(self, state):
        """Select an action based on the policy."""
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        distribution = Categorical(probs)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)

    def calculate_returns(self, rewards):
        """Calculate discounted returns."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns

    def train(self, num_episodes=2000, print_interval=100):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            rewards, log_probs = [], []
            episode_reward = 0
            done = False

            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                episode_reward += reward
                state = next_state

            returns = self.calculate_returns(rewards)
            loss = -torch.stack(log_probs) * returns
            loss = loss.sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.episode_rewards.append(episode_reward)
            rolling_avg = np.mean(self.episode_rewards[-self.window_size:])
            self.rolling_avg_rewards.append(rolling_avg)

            if self.convergence_episode is None and self.check_convergence():
                self.convergence_episode = episode + 1

            if (episode + 1) % print_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-print_interval:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")

    def plot_training_progress(self):
        """Plot training progress with convergence metrics."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, label="Episode Rewards")
        plt.plot(self.rolling_avg_rewards, label=f"Rolling Avg (window={self.window_size})", linewidth=2)
        if self.convergence_episode:
            plt.axvline(self.convergence_episode, color='r', linestyle='--', label="Convergence Point")
        plt.title("Training Progress")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate(self, num_episodes=10, render=False):
        eval_rewards = []
        env = gym.make('LunarLander-v2', render_mode='human') if render else self.env

        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                state = torch.FloatTensor(state)
                with torch.no_grad():
                    probs = self.policy(state)
                action = torch.argmax(probs).item()
                state, reward, done, _, _ = env.step(action)
                episode_reward += reward
            eval_rewards.append(episode_reward)

        avg_reward = np.mean(eval_rewards)
        print(f"Evaluation Average Reward: {avg_reward:.2f}")

    def save_policy(self, path='reinforce_lunarlander.pth'):
        torch.save(self.policy.state_dict(), path)

    def load_policy(self, path='reinforce_lunarlander.pth'):
        self.policy.load_state_dict(torch.load(path))


def main():
    agent = REINFORCE(env_name="LunarLander-v2")
    agent.train(num_episodes=2000)
    if agent.convergence_episode:
        print(f"Convergence achieved at episode {agent.convergence_episode}.")
    else:
        print("Convergence not achieved.")
    agent.plot_training_progress()
    agent.evaluate(num_episodes=10, render=False)


if __name__ == "__main__":
    main()