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
    def __init__(self, env_name='LunarLander-v3', hidden_dim=256, learning_rate=0.0005, gamma=0.99, window_size=100, tolerance=1.0):
        self.env = gym.make(env_name)
        self.input_dim = self.env.observation_space.shape[0]  # 8 for LunarLander
        self.output_dim = self.env.action_space.n  # 4 discrete actions
        self.gamma = gamma
        
        # Initialize Actor-Critic network
        self.actor_critic = ActorCriticNetwork(self.input_dim, hidden_dim, self.output_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # For tracking training progress
        self.episode_rewards = []
        self.rolling_avg_rewards = []
        self.convergence_episode = None
        self.window_size = window_size  # Rolling average window size
        self.tolerance = tolerance  # Convergence tolerance
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        policy_probs, _ = self.actor_critic(state)
        distribution = Categorical(policy_probs)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)
    
    def calculate_advantages(self, rewards, values, next_value, dones):
        returns = []
        G = next_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns)
        advantages = returns - values
        return returns, advantages
    
    def check_convergence(self):
        """Check if the rolling average has converged."""
        if len(self.rolling_avg_rewards) >= self.window_size:
            recent_avg = np.mean(self.rolling_avg_rewards[-self.window_size:])
            if abs(recent_avg - self.rolling_avg_rewards[-1]) <= self.tolerance:
                return True
        return False
    
    def train(self, num_episodes=2000, print_interval=100):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            rewards, log_probs, values, dones = [], [], [], []
            episode_reward = 0
            
            while True:
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                _, value = self.actor_critic(torch.FloatTensor(state))
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(done)
                episode_reward += reward
                
                if done:
                    _, next_value = self.actor_critic(torch.FloatTensor(next_state))
                    break
                
                state = next_state
            
            values = torch.cat(values).squeeze()
            returns, advantages = self.calculate_advantages(rewards, values, next_value.item(), dones)
            
            policy_loss = []
            value_loss = []
            for log_prob, advantage, value, ret in zip(log_probs, advantages, values, returns):
                policy_loss.append(-log_prob * advantage)
                value_loss.append((value - ret) ** 2)
            
            policy_loss = torch.stack(policy_loss).sum()
            value_loss = torch.stack(value_loss).sum()
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.episode_rewards.append(episode_reward)
            rolling_avg = np.mean(self.episode_rewards[-self.window_size:])
            self.rolling_avg_rewards.append(rolling_avg)
            
            # Check for convergence
            if self.convergence_episode is None and self.check_convergence():
                self.convergence_episode = episode + 1
            
            if (episode + 1) % print_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-print_interval:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    def plot_training_progress(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, label='Episode Reward')
        plt.plot(self.rolling_avg_rewards, label=f'Rolling Avg (window={self.window_size})', linewidth=2)
        if self.convergence_episode:
            plt.axvline(self.convergence_episode, color='r', linestyle='--', label='Convergence Point')
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate(self, num_episodes=10, render=False):
        env = gym.make('LunarLander-v3', render_mode='human') if render else self.env
        eval_rewards = []
        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                state = torch.FloatTensor(state)
                with torch.no_grad():
                    policy_probs, _ = self.actor_critic(state)
                action = torch.argmax(policy_probs).item()
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            eval_rewards.append(episode_reward)
        avg_reward = np.mean(eval_rewards)
        print(f"\nEvaluation over {num_episodes} episodes: {avg_reward:.2f}")
        return avg_reward
    
    def save_policy(self, path='a2c_lunarlander.pth'):
        torch.save(self.actor_critic.state_dict(), path)
    
    def load_policy(self, path='a2c_lunarlander.pth'):
        self.actor_critic.load_state_dict(torch.load(path))

def main():
    agent = A2C(hidden_dim=256, learning_rate=0.0005)
    agent.train(num_episodes=2000)
    if agent.convergence_episode:
        print(f"Convergence achieved at episode: {agent.convergence_episode}")
    else:
        print("Convergence not achieved during training.")
    agent.plot_training_progress()
    agent.evaluate(num_episodes=10, render=False)
    agent.save_policy()

if __name__ == "__main__":
    main()