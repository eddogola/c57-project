import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
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

    def forward_actor(self, x):
        return self.actor(self.shared(x))

    def forward_critic(self, x):
        return self.critic(self.shared(x))

    def forward(self, x):
        shared = self.shared(x)
        policy_probs = self.actor(shared)
        value = self.critic(shared)
        return policy_probs, value


class TRPO:
    def __init__(self, env_name='LunarLander-v2', hidden_dim=128, learning_rate=0.01, max_d_kl=0.01, window_size=100, gradient_threshold=0.005):
        self.env = gym.make(env_name)
        self.input_dim = self.env.observation_space.shape[0]  # 8 for LunarLander
        self.output_dim = self.env.action_space.n  # 4 discrete actions
        self.max_d_kl = max_d_kl
        self.learning_rate = learning_rate

        self.actor_critic = ActorCriticNetwork(self.input_dim, hidden_dim, self.output_dim)
        self.episode_rewards = []
        self.rolling_avg_rewards = []
        self.window_size = window_size
        self.gradient_threshold = gradient_threshold
        self.convergence_episode = None

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []

    def append_info(self, state, action, reward, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def calculate_advantages(self, rewards, values, next_value):
        """Calculate discounted returns and advantages."""
        returns = []
        G = next_value
        for r in reversed(rewards):
            G = r + 0.99 * G
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

    def surrogate_loss(self, new_probs, old_probs, advantages):
        return (new_probs / old_probs * advantages).mean()

    def kl_divergence(self, p, q):
        p = p.detach()
        return (p * (p.log() - q.log())).sum(-1).mean()

    def flat_grad(self, y, x, retain_graph=False, create_graph=False):
        grads = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        return torch.cat([g.view(-1) for g in grads])

    def conjugate_gradient(self, A, b, max_iterations=10, delta=0.0):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()

        for _ in range(max_iterations):
            AVP = A(p)
            alpha = (r @ r) / (p @ AVP)
            x_new = x + alpha * p

            if torch.norm(x_new - x) <= delta:
                return x_new

            r_new = r - alpha * AVP
            beta = (r_new @ r_new) / (r @ r)
            p = r_new + beta * p
            r = r_new
            x = x_new
        return x

    def apply_update(self, grad):
        """Apply policy gradient updates."""
        start = 0
        for param in self.actor_critic.actor.parameters():
            end = start + param.numel()
            param_update = grad[start:end].view(param.shape)
            param.data += param_update
            start = end

    def update_agent(self):
        """Trust Region Policy Optimization (TRPO) updates."""
        states = torch.cat(self.states, dim=0)
        actions = torch.cat(self.actions, dim=0).flatten()

        policy_probs, values = self.actor_critic(states)
        values = values.squeeze()
        advantages = self.calculate_advantages(self.rewards, values, torch.zeros(1))

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_probs = policy_probs.gather(1, actions.unsqueeze(-1)).squeeze()
        L = self.surrogate_loss(old_probs, old_probs.detach(), advantages)

        parameters = list(self.actor_critic.actor.parameters())
        g = self.flat_grad(L, parameters, retain_graph=True)

        d_kl = self.kl_divergence(policy_probs, policy_probs)
        HVP = lambda v: self.flat_grad(torch.dot(d_kl, v), parameters)

        search_dir = self.conjugate_gradient(HVP, g)
        max_step = search_dir @ HVP(search_dir)
        max_step = torch.sqrt(2 * self.max_d_kl / max_step)
        grad_step = max_step * search_dir

        self.apply_update(grad_step)

    def train(self, num_episodes=1000, print_interval=100):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            rewards = []
            episode_reward = 0
            self.states, self.actions, self.rewards = [], [], []

            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    policy_probs, _ = self.actor_critic(state_tensor)
                action = Categorical(policy_probs).sample()

                next_state, reward, done, _, _ = self.env.step(action.item())
                rewards.append(reward)

                self.append_info(state_tensor, torch.tensor(action), reward, torch.FloatTensor(next_state))
                state = next_state
                episode_reward += reward

            self.update_agent()
            self.episode_rewards.append(episode_reward)
            self.rolling_avg_rewards.append(np.mean(self.episode_rewards[-self.window_size:]))

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
        """Evaluate the trained policy."""
        env = gym.make('LunarLander-v2', render_mode='human') if render else self.env
        eval_rewards = []
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
        avg_reward = np.mean(eval_rewards)
        print(f"\nEvaluation over {num_episodes} episodes: {avg_reward:.2f}")

    def save_policy(self, path='trpo_lunarlander.pth'):
        torch.save(self.actor_critic.state_dict(), path)

    def load_policy(self, path='trpo_lunarlander.pth'):
        self.actor_critic.load_state_dict(torch.load(path))


def main():
    agent = TRPO()
    agent.train(num_episodes=1000)
    if agent.convergence_episode:
        print(f"Convergence achieved at episode {agent.convergence_episode}.")
    else:
        print("Convergence not achieved.")
    agent.plot_training_progress()
    agent.evaluate(num_episodes=10, render=False)


if __name__ == "__main__":
    main()