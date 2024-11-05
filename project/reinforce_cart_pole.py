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
    def __init__(self, env_name='CartPole-v1', hidden_dim=128, learning_rate=0.01, gamma=0.99):
        self.env = gym.make(env_name)
        # self.env = gym.make(env_name, render_mode="human")
        self.input_dim = self.env.observation_space.shape[0]  # 4 for CartPole
        self.output_dim = self.env.action_space.n  # 2 for CartPole
        self.gamma = gamma
        
        # Initialize policy network
        self.policy = PolicyNetwork(self.input_dim, hidden_dim, self.output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # For storing episode information
        self.rewards = []
        self.action_probs = []
        
        # For tracking training progress
        self.episode_rewards = []
        
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        distribution = Categorical(probs)
        action = distribution.sample()
        
        # Store log probability for training
        self.action_probs.append(distribution.log_prob(action))
        
        return action.item()
    
    def calculate_returns(self, rewards):
        """Calculate discounted returns for each timestep"""
        returns = []
        G = 0
        
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
            
        returns = torch.tensor(returns)
        # Normalize returns for stable training
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        return returns
    
    def update_policy(self):
        """Update policy network using REINFORCE algorithm"""
        returns = self.calculate_returns(self.rewards)
        policy_loss = []
        
        # Calculate policy loss
        for log_prob, G in zip(self.action_probs, returns):
            policy_loss.append(-log_prob * G)
        
        # Backpropagation
        policy_loss = torch.stack(policy_loss).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode memory
        self.rewards = []
        self.action_probs = []
        
        return policy_loss.item()
    
    def train(self, num_episodes=1000, print_interval=100):
        """Train the agent"""
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            
            # Run one episode
            while True:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.rewards.append(reward)
                episode_reward += reward
                
                if done:
                    break
                    
                state = next_state
            
            # Update policy after episode
            policy_loss = self.update_policy()
            self.episode_rewards.append(episode_reward)
            
            # Print progress
            if (episode + 1) % print_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-print_interval:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    def evaluate(self, num_episodes=10, render=False):
        """Evaluate the trained policy"""
        if render:
            env = gym.make('CartPole-v1', render_mode='human')
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
                    probs = self.policy(state)
                action = torch.argmax(probs).item()  # Use greedy action selection
                
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
        
        avg_reward = np.mean(eval_rewards)
        print(f"\nEvaluation over {num_episodes} episodes: {avg_reward:.2f}")
        return avg_reward
    
    def plot_training_progress(self):
        """Plot the training progress"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards)
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.grid(True)
        plt.show()
    
    def save_policy(self, path='reinforce_policy.pth'):
        """Save the policy network"""
        torch.save(self.policy.state_dict(), path)
    
    def load_policy(self, path='reinforce_policy.pth'):
        """Load a saved policy network"""
        self.policy.load_state_dict(torch.load(path))

def main():
    # Create and train the agent
    agent = REINFORCE(hidden_dim=128, learning_rate=0.01)
    agent.train(num_episodes=1000)
    
    # Plot training progress
    agent.plot_training_progress()
    
    # Evaluate the trained agent
    agent.evaluate(num_episodes=10, render=True)
    
    # Save the trained policy
    agent.save_policy()

if __name__ == "__main__":
    main()