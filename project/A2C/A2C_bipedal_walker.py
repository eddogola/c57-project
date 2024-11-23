import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        # Shared network for feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Actor (Gaussian policy: mean and log_std)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log std
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single value output for V(s)
        )
    
    def forward(self, x):
        shared_output = self.shared(x)
        mean = self.actor_mean(shared_output)
        std = torch.exp(self.actor_log_std)
        value = self.critic(shared_output)
        return mean, std, value

class A2C:
    def __init__(self, env_name='BipedalWalker-v3', hidden_dim=256, learning_rate=0.0003, gamma=0.99):
        self.env = gym.make(env_name)
        self.input_dim = self.env.observation_space.shape[0]  # 24 for BipedalWalker
        self.action_dim = self.env.action_space.shape[0]  # 4 continuous actions
        self.gamma = gamma
        
        # Initialize Actor-Critic network
        self.actor_critic = ActorCriticNetwork(self.input_dim, hidden_dim, self.action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # For tracking training progress
        self.episode_rewards = []
    
    def select_action(self, state):
        """Select an action based on the Gaussian policy."""
        state = torch.FloatTensor(state)
        mean, std, _ = self.actor_critic(state)
        distribution = Normal(mean, std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1)  # Sum log probs for multidimensional action
        return action.detach().numpy(), log_prob
    
    def calculate_advantages(self, rewards, values, next_value, dones):
        """Calculate discounted returns and advantages."""
        returns = []
        G = next_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns)
        advantages = returns - values
        return returns, advantages
    
    def train(self, num_episodes=1000, print_interval=100):
        """Train the A2C agent."""
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            rewards, log_probs, values, dones = [], [], [], []
            episode_reward = 0
            
            # Run one episode
            while True:
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store episode data
                _, _, value = self.actor_critic(torch.FloatTensor(state))
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                dones.append(done)
                episode_reward += reward
                
                if done:
                    _, _, next_value = self.actor_critic(torch.FloatTensor(next_state))
                    break
                
                state = next_state
            
            # Update policy and value network
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
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.episode_rewards.append(episode_reward)
            
            # Print progress
            if (episode + 1) % print_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-print_interval:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    def evaluate(self, num_episodes=10, render=False):
        """Evaluate the trained policy."""
        if render:
            env = gym.make('BipedalWalker-v3', render_mode='human')
        else:
            env = self.env
            
        eval_rewards = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                state = torch.FloatTensor(state)
                with torch.no_grad():
                    mean, _, _ = self.actor_critic(state)
                action = mean.numpy()  # Use the mean action for evaluation
                
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
        
        avg_reward = np.mean(eval_rewards)
        print(f"\nEvaluation over {num_episodes} episodes: {avg_reward:.2f}")
        return avg_reward
    
    def plot_training_progress(self):
        """Plot the training progress."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards)
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.grid(True)
        plt.show()
    
    def save_policy(self, path='a2c_bipedalwalker.pth'):
        """Save the policy network."""
        torch.save(self.actor_critic.state_dict(), path)
    
    def load_policy(self, path='a2c_bipedalwalker.pth'):
        """Load a saved policy network."""
        self.actor_critic.load_state_dict(torch.load(path))

def main():
    # Create and train the agent
    agent = A2C(hidden_dim=256, learning_rate=0.0003)
    agent.train(num_episodes=2000)
    
    # Plot training progress
    agent.plot_training_progress()
    
    # Evaluate the trained agent
    agent.evaluate(num_episodes=10, render=True)
    
    # Save the trained policy
    agent.save_policy()

if __name__ == "__main__":
    main()
