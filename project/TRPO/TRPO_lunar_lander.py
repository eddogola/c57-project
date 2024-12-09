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
            nn.Softmax(dim=1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single value output for V(s)
        )
    
    def forward_actor(self, x):
        """Forward pass for the policy (actor)."""
        return self.actor(self.shared(x))
    
    def forward_critic(self, x):
        """Forward pass for the value function (critic)."""
        return self.critic(self.shared(x))

    def forward(self, x):
        shared = self.shared(x)
        policy_probs = self.actor(shared)
        value = self.critic(shared)
        return policy_probs, value


class TRPO:
    def __init__(self, env_name='LunarLander-v3', hidden_dim=128, learning_rate=0.01, max_d_kl = 0.01):
        self.env = gym.make(env_name)
        self.input_dim = self.env.observation_space.shape[0]   # 8 for LunarLander
        self.output_dim = self.env.action_space.n  # 4 discrete actions
        self.max_d_kl = max_d_kl

        # stuff for network
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        
        # Initialize policy network
        self.actor_critic = ActorCriticNetwork(self.input_dim, hidden_dim, self.output_dim)
        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.actor_critic.critic.parameters(), lr=learning_rate)
        # For tracking training progress
        self.episode_rewards = []
        
    def append_info(self, state, action, reward, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def select_action(self, state):
        """Select action using current policy"""
        state = torch.tensor(state).float().unsqueeze(0)  # Turn state into a batch with a single element
        probs, _ = self.actor_critic(state)
        distribution = Categorical(probs)  # Create a distribution from probabilities for actions
        return distribution.sample().item()

    def update_critic(self, advantages):
        loss = .5 * (advantages ** 2).mean()  # MSE
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
    
    def estimate_advantages(self, states, last_state, rewards):
        _, values = self.actor_critic(states)
        _, last_value = self.actor_critic(last_state.unsqueeze(0))
        next_values = torch.zeros_like(rewards)
        for i in reversed(range(rewards.shape[0])):
            last_value = next_values[i] = rewards[i] + 0.99 * last_value
        advantages = next_values - values
        return advantages
    

    # Derivative stuff :)

    def surrogate_loss(self, new_probabilities, old_probabilities, advantages):
        return (new_probabilities / old_probabilities * advantages).mean()


    def kl_div(self, p, q):
        p = p.detach()
        return (p * (p.log() - q.log())).sum(-1).mean()


    def flat_grad(self, y, x, retain_graph=False, create_graph=False):
        if create_graph:
            retain_graph = True

        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = torch.cat([t.view(-1) for t in g])
        return g


    def conjugate_gradient(self, A, b, delta=0., max_iterations=10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()

        i = 0
        while i < max_iterations:
            AVP = A(p)

            dot_old = r @ r
            alpha = dot_old / (p @ AVP)

            x_new = x + alpha * p

            if (x - x_new).norm() <= delta:
                return x_new

            i += 1
            r = r - alpha * AVP

            beta = (r @ r) / dot_old
            p = r + beta * p

            x = x_new
        return x

    def apply_update(self, grad_flattened):
        n = 0
        for p in self.actor_critic.actor.parameters():
            numel = p.numel()
            g = grad_flattened[n:n + numel].view(p.shape)
            p.data += g
            n += numel

    def update_agent(self):
        states = torch.cat([state for state in self.states], dim=0)
        actions = torch.cat([action for action in self.actions], dim=0).flatten()

        advantages = [self.estimate_advantages(self.states[i], self.next_states[i][-1], self.rewards[i]) for i in range(len(self.rewards))]
        advantages = torch.cat(advantages, dim=0).flatten()

        # Normalize advantages to reduce skewness and improve convergence
        advantages = (advantages - advantages.mean()) / advantages.std()

        self.update_critic(advantages)
        distribution, _ = self.actor_critic(states)
        distribution = torch.distributions.utils.clamp_probs(distribution)
        probabilities = distribution[range(distribution.shape[0]), actions]

        # Now we have all the data we need for the algorithm

        # We will calculate the gradient wrt to the new probabilities (surrogate function),
        # so second probabilities should be treated as a constant
        L = self.surrogate_loss(probabilities, probabilities.detach(), advantages)
        KL = self.kl_div(distribution, distribution)

        parameters = list(self.actor_critic.actor.parameters())

        g = self.flat_grad(L, parameters, retain_graph=True)
        d_kl = self.flat_grad(KL, parameters, create_graph=True)  # Create graph, because we will call backward() on it (for HVP)

        def HVP(v):
            return self.flat_grad(d_kl @ v, parameters, retain_graph=True)

        search_dir = self.conjugate_gradient(HVP, g)
        max_length = torch.sqrt(2 * self.max_d_kl / (search_dir @ HVP(search_dir)))
        max_step = max_length * search_dir

        def criterion(step):
            self.apply_update(step)

            with torch.no_grad():
                distribution_new, _ = self.actor_critic(states)
                distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
                probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

                L_new = self.surrogate_loss(probabilities_new, probabilities, advantages)
                KL_new = self.kl_div(distribution, distribution_new)

            L_improvement = L_new - L

            if L_improvement > 0 and KL_new <= self.max_d_kl:
                return True

            self.apply_update(-step)
            return False

        i = 0
        while not criterion((0.9 ** i) * max_step) and i < 10:
            i += 1

    def train(self, num_episodes=1000, print_interval=100):
        """Train the agent"""
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False

            samples = []

            episode_reward = 0

            while not done:
                with torch.no_grad():
                    action = self.select_action(state)

                next_state, reward, done, _, _ = self.env.step(action)

                # Collect samples
                samples.append((state, action, reward, next_state))
                episode_reward += reward
                state = next_state

            # Transpose our samples
            states, actions, rewards, next_states = zip(*samples)

            states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
            next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
            actions = torch.as_tensor(actions).unsqueeze(1)
            rewards = torch.as_tensor(rewards).unsqueeze(1)


            self.append_info(states, actions, rewards, next_states)
            self.update_agent()

            self.episode_rewards.append(episode_reward)

            if (episode + 1) % print_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-print_interval:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    

    def evaluate(self, num_episodes=10, render=False):
        """Evaluate the trained policy"""
        if render:
            env = gym.make('LunarLander-v3', render_mode='human')
        else:
            env = self.env
            
        eval_rewards = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                state = torch.tensor(state).float().unsqueeze(0)  # Turn state into a batch with a single element
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

    def plot_training_progress(self, path = None):
        """Plot the training progress"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards)
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.grid(True)
        plt.show()
        # if path is not None:
        #     plt.savefig()

    def save_policy(self, path='trpo_policy_lunar_lander.pth'):
        """Save the policy network"""
        torch.save(self.actor_critic.state_dict(), path)

    def load_policy(self, path='trpo_policy_lunar_lander.pth'):
        """Load a saved policy network"""
        self.actor_critic.load_state_dict(torch.load(path))

def main():
    # Create and train the agent
    # Create and train the agent
    lr = 0.01
    d_kl = 0.001
    agent = TRPO(hidden_dim=128, learning_rate=lr, max_d_kl=d_kl)
    agent.train(num_episodes=1000)
    
    # Plot training progress
    agent.plot_training_progress(path=f"../reward graphs/TRPO/trpo_lunar_lander_{d_kl}_{lr}_2")
    
    # Save the trained policy
    agent.save_policy(path=f"trpo_policy_lunar_lander_{d_kl}_{lr}_2")

    # Evaluate the trained agent
    agent.evaluate(num_episodes=10, render=True)
    
    

if __name__ == "__main__":
    main()

    # learning_rate=0.01, max_d_kl=0.001
        # trpo_policy_lunar_lander_0_001
        # Episode 100, Average Reward: -105.48
        # Episode 200, Average Reward: -154.44
        # Episode 300, Average Reward: -206.23
        # Episode 400, Average Reward: -82.33
        # Episode 500, Average Reward: -81.98
        # Episode 600, Average Reward: -76.21
        # Episode 700, Average Reward: -92.78
        # Episode 800, Average Reward: -154.74
        # Episode 900, Average Reward: -88.39
        # Episode 1000, Average Reward: -68.60

        # Evaluation over 10 episodes: -7.50

        # take 2
            # Episode 100, Average Reward: -120.13
            # Episode 200, Average Reward: -118.93
            # Episode 300, Average Reward: -70.71
            # Episode 400, Average Reward: -91.91
            # Episode 500, Average Reward: -74.62
            # Episode 600, Average Reward: -153.01
            # Episode 700, Average Reward: -115.81
            # Episode 800, Average Reward: -120.99
            # Episode 900, Average Reward: -95.68
            # Episode 1000, Average Reward: -94.17

            # Evaluation over 10 episodes: -106.39
            
        #take 3
            # Episode 100, Average Reward: -116.62
            # Episode 200, Average Reward: -109.14
            # Episode 300, Average Reward: -188.12
            # Episode 400, Average Reward: -90.90
            # Episode 500, Average Reward: -75.31
            # Episode 600, Average Reward: -61.43
            # Episode 700, Average Reward: -66.53
            # Episode 800, Average Reward: -59.20
            # Episode 900, Average Reward: -79.10
            # Episode 1000, Average Reward: -73.92

            # Evaluation over 10 episodes: -84.81

        # FINAL

            # Episode 100, Average Reward: -308.09
            # Episode 200, Average Reward: -330.71
            # Episode 300, Average Reward: -273.56
            # Episode 400, Average Reward: -114.36
            # Episode 500, Average Reward: -89.45
            # Episode 600, Average Reward: -84.24
            # Episode 700, Average Reward: -80.64
            # Episode 800, Average Reward: -75.64
            # Episode 900, Average Reward: -77.76
            # Episode 1000, Average Reward: -71.57

            # Evaluation over 10 episodes: -33.37


    # lr = 0.005
    # d_kl = 0.001
        # Episode 100, Average Reward: -175.84
        # Episode 200, Average Reward: -133.50
        # Episode 300, Average Reward: -180.00
        # Episode 400, Average Reward: -179.46
        # Episode 500, Average Reward: -212.85
        # Episode 600, Average Reward: -205.71
        # Episode 700, Average Reward: -57.38
        # Episode 800, Average Reward: -87.41
        # Episode 900, Average Reward: -141.61
        # Episode 1000, Average Reward: -216.80

        # Evaluation over 10 episodes: -337.63