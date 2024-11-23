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
            nn.Linear(hidden_dim, 1)  # Single value output for V(s)
        )
    
    def forward(self, x):
        shared_output = self.shared(x)
        policy_probs = self.actor(shared_output)
        value = self.critic(shared_output)
        return policy_probs, value

class A2C:
    def __init__(self, env_name='CartPole-v1', hidden_dim=128, learning_rate=0.01, gamma=0.99):
        self.env = gym.make(env_name)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n
        self.gamma = gamma
        
        # Initialize Actor-Critic network
        self.actor_critic = ActorCriticNetwork(self.input_dim, hidden_dim, self.output_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # For tracking training progress
        self.episode_rewards = []
        
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state)
        policy_probs, _ = self.actor_critic(state)
        distribution = Categorical(policy_probs)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)
    
    def calculate_advantages(self, rewards, values, next_value, dones):
        """Calculate the advantages and targets"""
        returns = []
        G = next_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.tensor(returns)
        advantages = returns - values
        return returns, advantages
    
    def train(self, num_episodes=1000, print_interval=100):
        """Train the agent using A2C"""
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            rewards = []
            log_probs = []
            values = []
            dones = []
            episode_reward = 0
            
            # Run one episode
            while True:
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store episode data
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
                    policy_probs, _ = self.actor_critic(state)
                action = torch.argmax(policy_probs).item()  # Use greedy action selection
                
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
    
    def save_policy(self, path='a2c_policy.pth'):
        """Save the policy network"""
        torch.save(self.actor_critic.state_dict(), path)
    
    def load_policy(self, path='a2c_policy.pth'):
        """Load a saved policy network"""
        self.actor_critic.load_state_dict(torch.load(path))

def main():
    # Create and train the agent
    agent = A2C(hidden_dim=128, learning_rate=0.01)
    agent.train(num_episodes=1000)
    
    # Plot training progress
    agent.plot_training_progress()
    
    # Evaluate the trained agent
    agent.evaluate(num_episodes=10, render=True)
    
    # Save the trained policy
    agent.save_policy()

if __name__ == "__main__":
    main()
