import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import random
import math
import numpy as np
from collections import namedtuple

# Import the controller from the model file
from model import DQNController

# Define the structure for a single experience tuple
Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward'))

class PrioritizedReplayBuffer:
    """
    A Prioritized Experience Replay (PER) buffer.
    It stores experiences and samples them based on their TD-error, allowing
    the agent to learn more from "surprising" or important events.
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # Controls how much prioritization is used
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.epsilon = 1e-6 # Small constant to ensure no priority is zero

    def push(self, *args):
        """Saves a new experience with maximum priority to ensure it gets sampled at least once."""
        # Validate state shape consistency
        state = args[0]
        next_state = args[2]
        
        if not isinstance(state, torch.Tensor) or state.dim() != 1:
            raise ValueError(f"State must be a 1D tensor, got shape {state.shape if isinstance(state, torch.Tensor) else type(state)}")
        if not isinstance(next_state, torch.Tensor) or next_state.dim() != 1:
            raise ValueError(f"Next state must be a 1D tensor, got shape {next_state.shape if isinstance(next_state, torch.Tensor) else type(next_state)}")
        
        max_priority = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(Experience(*args))
        else:
            self.memory[self.position] = Experience(*args)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Samples a batch of experiences using prioritized sampling.
        Also calculates importance-sampling (IS) weights to correct for the bias.
        """
        if len(self.memory) == 0:
            return [], [], []

        priorities = self.priorities[:len(self.memory)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        experiences = [self.memory[i] for i in indices]
        
        # --- Calculate Importance-Sampling (IS) Weights ---
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max() # Normalize for stability
        weights = torch.tensor(weights, dtype=torch.float32)

        return experiences, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """Updates the priorities of sampled experiences after a learning step."""
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority + self.epsilon

    def __len__(self):
        return len(self.memory)


class DQNTrainer:
    """
    The DQN Trainer, upgraded with Double DQN and Prioritized Experience Replay.
    """
    def __init__(self, model, block_flops, block_params, args):
        self.model = model
        self.args = args
        self.device = next(model.parameters()).device
        
        self.policy_net = model.controller
        self.target_net = DQNController(
            feature_dim=args.num_features,
            hidden_dim=args.controller_hidden_dim,
            num_actions=model.action_space_size
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.controller_optimizer = optim.AdamW(self.policy_net.parameters(), lr=args.controller_lr)
        self.worker_optimizer = optim.AdamW(self.model.get_worker_params(), lr=args.worker_lr)

        # Controller LR scheduler: cosine annealing with warm restarts
        # T_0 = restart period in epochs, T_mult = multiplier after each restart,
        # eta_min = minimum LR floor
        ctrl_T0 = getattr(args, 'ctrl_scheduler_T0', 20)
        ctrl_Tmult = getattr(args, 'ctrl_scheduler_Tmult', 2)
        ctrl_eta_min = getattr(args, 'ctrl_scheduler_eta_min', 1e-5)
        self.controller_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            self.controller_optimizer,
            T_0=ctrl_T0,
            T_mult=ctrl_Tmult,
            eta_min=ctrl_eta_min
        )

        # Initialize the Prioritized Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_capacity, alpha=args.per_alpha)

        self.block_flops = block_flops
        self.block_params = block_params
        
        self.steps_done = 0
        # Beta parameter for PER, will be annealed from beta_start to 1.0
        self.beta = args.per_beta_start

    def select_action(self, state):
        eps_threshold = self.args.eps_end + (self.args.eps_start - self.args.eps_end) * \
                        math.exp(-1. * self.steps_done / self.args.eps_decay)
        self.steps_done += 1
        
        if random.random() < eps_threshold:
            return torch.tensor([[random.randrange(self.model.action_space_size)]], 
                                device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                q_values, _ = self.policy_net(state)
                return q_values.max(1)[1].view(1, 1)

    def optimize_controller(self):
        if len(self.replay_buffer) < self.args.batch_size:
            return

        # Sample from prioritized replay buffer
        experiences, indices, weights = self.replay_buffer.sample(
            self.args.batch_size, 
            beta=self.beta
        )
        batch = Experience(*zip(*experiences))

        # Stack states (each state is [num_features])
        # Pad states to same size before stacking
        max_size = max(s.size(0) for s in batch.state)
        state_batch = torch.stack([
            torch.nn.functional.pad(s, (0, max_size - s.size(0))) for s in batch.state
        ]).to(self.device)
        next_state_batch = torch.stack([
            torch.nn.functional.pad(s, (0, max_size - s.size(0))) for s in batch.next_state
        ]).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        weights = weights.to(self.device)

        state_action_values = self.policy_net(state_batch)[0].gather(1, action_batch)

        # --- DOUBLE DQN LOGIC ---
        with torch.no_grad():
            best_next_actions = self.policy_net(next_state_batch)[0].max(1)[1].unsqueeze(1)
            next_state_values = self.target_net(next_state_batch)[0].gather(1, best_next_actions).squeeze(1)

        expected_state_action_values = (next_state_values * self.args.gamma) + reward_batch

        # --- PER: Update priorities with the new TD-error ---
        td_error = (expected_state_action_values.unsqueeze(1) - state_action_values).abs().detach()
        self.replay_buffer.update_priorities(indices, td_error.cpu().numpy().flatten())

        # --- PER: Compute the weighted loss ---
        loss = (F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none') * weights.to(self.device).unsqueeze(1)).mean()
        
        self.controller_optimizer.zero_grad()
        loss.backward()
        if self.args.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.controller_optimizer.step()

    def step_controller_scheduler(self, epoch):
        """Step the controller LR scheduler. Call once per epoch."""
        self.controller_scheduler.step(epoch)

    def get_controller_lr(self):
        """Return the current controller learning rate."""
        return self.controller_optimizer.param_groups[0]['lr']

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())