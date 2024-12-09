import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorCriticNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        shared_output = self.shared(x)
        policy_probs = self.actor(shared_output)
        value = self.critic(shared_output)
        return policy_probs, value


class A2C:
    def __init__(self, env_name='CartPole-v1', hidden_dim=256, learning_rate=0.001, gamma=0.99, window_size=100, gradient_threshold=0.001):
        self.env = gym.make(env_name)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n
        self.gamma = gamma

        self.actor_critic = ActorCriticNetwork(self.input_dim, hidden_dim, self.output_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        self.episode_rewards = []
        self.rolling_avg_rewards = []
        self.window_size = window_size
        self.gradient_threshold = gradient_threshold
        self.convergence_episode = None

    def calculate_advantages(self, rewards, values, next_value, dones):
        """Calculate discounted returns and advantages."""
        returns = []
        G = next_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = returns - values
        return returns, advantages

    def calculate_gradient(self):
        """Calculate the gradient of the rolling average."""
        if len(self.rolling_avg_rewards) < self.window_size:
            return None
        x = np.arange(self.window_size)
        y = np.array(self.rolling_avg_rewards[-self.window_size:])
        gradient = np.polyfit(x, y, 1)[0]  # Fit a linear model and return its slope
        return gradient

    def check_convergence(self):
        """Check if the gradient of the rolling average is below the threshold."""
        if len(self.rolling_avg_rewards) < max(self.window_size, 200):  # Require a minimum number of episodes
            return False

        gradient = self.calculate_gradient()
        if gradient is not None and abs(gradient) < self.gradient_threshold:
            return True
        return False


    def train(self, num_episodes=1000, print_interval=100):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            rewards, log_probs, values, dones = [], [], [], []
            episode_reward = 0

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                policy_probs, value = self.actor_critic(state_tensor)
                distribution = Categorical(policy_probs)
                action = distribution.sample()

                next_state, reward, done, _, _ = self.env.step(action.item())
                rewards.append(reward)
                log_probs.append(distribution.log_prob(action))
                values.append(value)
                dones.append(done)
                episode_reward += reward
                state = next_state

            # Bootstrap value for the final state
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            _, next_value = self.actor_critic(next_state_tensor)
            returns, advantages = self.calculate_advantages(rewards, torch.cat(values).squeeze(), next_value.item(), dones)

            # Compute losses
            policy_loss = -(torch.stack(log_probs) * advantages.detach()).sum()
            value_loss = advantages.pow(2).mean()
            loss = policy_loss + value_loss

            # Optimize
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
        env = gym.make('CartPole-v1', render_mode='human') if render else self.env

        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    policy_probs, _ = self.actor_critic(state_tensor)
                    action = torch.argmax(policy_probs).item()
                state, reward, done, _, _ = env.step(action)
                episode_reward += reward
            eval_rewards.append(episode_reward)

        print(f"Evaluation Average Reward: {np.mean(eval_rewards):.2f}")

    def save_policy(self, path='a2c_cartpole.pth'):
        torch.save(self.actor_critic.state_dict(), path)

    def load_policy(self, path='a2c_cartpole.pth'):
        self.actor_critic.load_state_dict(torch.load(path))


def main():
    agent = A2C()
    agent.train(num_episodes=1000)
    if agent.convergence_episode:
        print(f"Convergence achieved at episode {agent.convergence_episode}.")
    else:
        print("Convergence not achieved.")
    agent.plot_training_progress()
    agent.evaluate(num_episodes=10, render=False)


if __name__ == "__main__":
    main()