import gymnasium as gym
import numpy as np

class QLearningCartPole:
    def __init__(self, n_bins=100, n_episodes=10000, max_steps=500):
        self.env = gym.make('CartPole-v1')
        # self.env = gym.make('CartPole-v1', render_mode="human")
        self.n_bins = n_bins
        self.n_episodes = n_episodes
        self.max_steps = max_steps

        # Define state space bins
        self.position_space = np.linspace(-2.4, 2.4, n_bins)
        self.velocity_space = np.linspace(-4, 4, n_bins)
        self.angle_space = np.linspace(-0.209, 0.209, n_bins)
        self.angular_velocity_space = np.linspace(-4, 4, n_bins)
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_bins, n_bins, n_bins, n_bins, 2))  # 2 actions
        
        # Learning parameters
        self.learning_rate = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Performance tracking
        self.rewards_history = []
        self.steps_history = []
        
    def discretize_state(self, state):
        """Convert continuous state to discrete state indices"""
        position, velocity, angle, angular_velocity = state
        
        position_bin = np.digitize(position, self.position_space)
        velocity_bin = np.digitize(velocity, self.velocity_space)
        angle_bin = np.digitize(angle, self.angle_space)
        angular_velocity_bin = np.digitize(angular_velocity, self.angular_velocity_space)
        
        return (position_bin-1, velocity_bin-1, angle_bin-1, angular_velocity_bin-1)
    
    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])
    
    def train(self):
        """Train the Q-learning agent"""
        for episode in range(self.n_episodes):
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            episode_reward = 0
            
            for step in range(self.max_steps):
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.discretize_state(next_state)
                done = terminated or truncated
                
                # Q-learning update
                old_value = self.q_table[state + (action,)]
                next_max = np.max(self.q_table[next_state])
                new_value = old_value + self.learning_rate * (
                    reward + self.gamma * next_max - old_value
                )
                self.q_table[state + (action,)] = new_value
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Track performance
            self.rewards_history.append(episode_reward)
            self.steps_history.append(step + 1)
            
            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.rewards_history[-100:])
                avg_steps = np.mean(self.steps_history[-100:])
                print(f"Episode: {episode + 1}, Avg Reward: {avg_reward:.2f}, "
                      f"Avg Steps: {avg_steps:.2f}, Epsilon: {self.epsilon:.3f}")
    
    def evaluate(self, n_eval_episodes=10):
        """Evaluate the trained agent"""
        eval_rewards = []
        eval_steps = []
        
        for episode in range(n_eval_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(self.max_steps):
                state = self.discretize_state(state)
                action = np.argmax(self.q_table[state])  # Always choose best action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_steps.append(step + 1)
        
        print("\nEvaluation Results:")
        print(f"Average Reward: {np.mean(eval_rewards):.2f}")
        print(f"Average Steps: {np.mean(eval_steps):.2f}")
        return np.mean(eval_rewards), np.mean(eval_steps)

def main():
    # Create and train the agent
    agent = QLearningCartPole(n_bins=10, n_episodes=1000)
    agent.train()
    
    # Evaluate the trained agent
    agent.evaluate()
    
    # Close the environment
    agent.env.close()

if __name__ == "__main__":
    main()