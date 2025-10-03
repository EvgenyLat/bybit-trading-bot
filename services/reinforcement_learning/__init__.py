"""
Reinforcement Learning Trading Agent
Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) for trading
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
import random
import gym
from gym import spaces
import yaml

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """Custom trading environment for RL"""
    
    def __init__(self, data: pd.DataFrame, config: Dict):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.config = config
        self.current_step = 0
        self.max_steps = len(data) - 1
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: features + portfolio state
        feature_dim = len(data.columns)
        portfolio_dim = 4  # cash, position, unrealized_pnl, realized_pnl
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(feature_dim + portfolio_dim,), dtype=np.float32
        )
        
        # Initialize portfolio
        self.initial_cash = config.get('initial_cash', 10000)
        self.cash = self.initial_cash
        self.position = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trade_history = []
        
    def reset(self):
        """Reset environment"""
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trade_history = []
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one step in environment"""
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, {}
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Calculate portfolio value
        portfolio_value = self.cash + self.position * current_price
        
        info = {
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }
        
        return self._get_observation(), reward, done, info
    
    def _execute_action(self, action: int, price: float) -> float:
        """Execute trading action"""
        reward = 0.0
        
        if action == 1:  # Buy
            if self.cash > price:
                shares_to_buy = self.cash / price
                self.position += shares_to_buy
                self.cash = 0.0
                self.trade_history.append(('buy', price, shares_to_buy))
                
        elif action == 2:  # Sell
            if self.position > 0:
                self.cash = self.position * price
                self.realized_pnl += self.position * price - self.position * self.trade_history[-1][1] if self.trade_history else 0
                self.position = 0.0
                self.trade_history.append(('sell', price, self.position))
        
        # Calculate reward based on portfolio performance
        portfolio_value = self.cash + self.position * price
        
        if len(self.trade_history) > 0:
            # Reward based on profit/loss
            if self.trade_history[-1][0] == 'sell':
                reward = self.realized_pnl / self.initial_cash
            else:
                # Reward based on unrealized P&L
                reward = (portfolio_value - self.initial_cash) / self.initial_cash
        
        return reward
    
    def _get_observation(self):
        """Get current observation"""
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape[0])
        
        # Market features
        market_features = self.data.iloc[self.current_step].values
        
        # Portfolio state
        portfolio_state = np.array([
            self.cash / self.initial_cash,  # Normalized cash
            self.position,  # Current position
            self.unrealized_pnl / self.initial_cash,  # Normalized unrealized P&L
            self.realized_pnl / self.initial_cash  # Normalized realized P&L
        ])
        
        return np.concatenate([market_features, portfolio_state]).astype(np.float32)


class DQNNetwork(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 3):
        super(DQNNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.lr = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.95)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 32)
        self.memory_size = config.get('memory_size', 10000)
        
        # Networks
        self.q_network = DQNNetwork(state_dim, output_dim=action_dim)
        self.target_network = DQNNetwork(state_dim, output_dim=action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Experience replay
        self.memory = deque(maxlen=self.memory_size)
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Hyperparameters
        self.lr = config.get('learning_rate', 0.0003)
        self.gamma = config.get('gamma', 0.99)
        self.eps_clip = config.get('eps_clip', 0.2)
        self.k_epochs = config.get('k_epochs', 4)
        self.batch_size = config.get('batch_size', 64)
        
        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Memory
        self.memory = PPOMemory()
    
    def act(self, state):
        """Choose action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        return action, action_probs[0][action].item()
    
    def update(self):
        """Update policy using PPO"""
        if len(self.memory.states) < self.batch_size:
            return
        
        # Calculate advantages
        rewards = self.memory.rewards
        dones = self.memory.dones
        
        advantages = []
        advantage = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                advantage = 0
            advantage = rewards[i] + self.gamma * advantage
            advantages.insert(0, advantage)
        
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(self.memory.states)
        actions = torch.LongTensor(self.memory.actions)
        old_probs = torch.FloatTensor(self.memory.probs)
        
        # Update policy
        for _ in range(self.k_epochs):
            action_probs = self.policy(states)
            new_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            ratio = new_probs / old_probs
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear memory
        self.memory.clear()


class PolicyNetwork(nn.Module):
    """Policy network for PPO"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
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


class PPOMemory:
    """Memory for PPO agent"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.dones = []
    
    def store(self, state, action, reward, prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.probs.append(prob)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.probs.clear()
        self.dones.clear()


class RLTrainingService:
    """Service for training RL agents"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def train_dqn_agent(self, data: pd.DataFrame, episodes: int = 1000) -> DQNAgent:
        """Train DQN agent"""
        try:
            logger.info("Starting DQN training")
            
            # Create environment
            env = TradingEnvironment(data, self.config)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            
            # Create agent
            agent = DQNAgent(state_dim, action_dim, self.config)
            
            # Training loop
            scores = []
            for episode in range(episodes):
                state = env.reset()
                total_reward = 0
                
                while True:
                    action = agent.act(state, training=True)
                    next_state, reward, done, info = env.step(action)
                    
                    agent.remember(state, action, reward, next_state, done)
                    agent.replay()
                    
                    state = next_state
                    total_reward += reward
                    
                    if done:
                        break
                
                scores.append(total_reward)
                
                # Update target network every 10 episodes
                if episode % 10 == 0:
                    agent.update_target_network()
                
                # Log progress
                if episode % 100 == 0:
                    avg_score = np.mean(scores[-100:])
                    logger.info(f"Episode {episode}, Average Score: {avg_score:.2f}")
            
            logger.info("DQN training completed")
            return agent
            
        except Exception as e:
            logger.error(f"Error training DQN agent: {e}")
            raise
    
    def train_ppo_agent(self, data: pd.DataFrame, episodes: int = 1000) -> PPOAgent:
        """Train PPO agent"""
        try:
            logger.info("Starting PPO training")
            
            # Create environment
            env = TradingEnvironment(data, self.config)
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            
            # Create agent
            agent = PPOAgent(state_dim, action_dim, self.config)
            
            # Training loop
            scores = []
            for episode in range(episodes):
                state = env.reset()
                total_reward = 0
                
                while True:
                    action, prob = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    
                    agent.memory.store(state, action, reward, prob, done)
                    
                    state = next_state
                    total_reward += reward
                    
                    if done:
                        break
                
                # Update agent
                agent.update()
                
                scores.append(total_reward)
                
                # Log progress
                if episode % 100 == 0:
                    avg_score = np.mean(scores[-100:])
                    logger.info(f"Episode {episode}, Average Score: {avg_score:.2f}")
            
            logger.info("PPO training completed")
            return agent
            
        except Exception as e:
            logger.error(f"Error training PPO agent: {e}")
            raise
    
    def evaluate_agent(self, agent, data: pd.DataFrame) -> Dict:
        """Evaluate trained agent"""
        try:
            env = TradingEnvironment(data, self.config)
            state = env.reset()
            total_reward = 0
            trades = []
            
            while True:
                action = agent.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                if action != 0:  # Not hold
                    trades.append({
                        'action': action,
                        'price': data.iloc[env.current_step]['close'],
                        'portfolio_value': info['portfolio_value']
                    })
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Calculate metrics
            final_value = info['portfolio_value']
            total_return = (final_value - self.config.get('initial_cash', 10000)) / self.config.get('initial_cash', 10000)
            
            return {
                'total_reward': total_reward,
                'final_value': final_value,
                'total_return': total_return,
                'num_trades': len(trades),
                'trades': trades
            }
            
        except Exception as e:
            logger.error(f"Error evaluating agent: {e}")
            return {}

