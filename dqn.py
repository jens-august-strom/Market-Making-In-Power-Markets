import os
import random
import time
import csv
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from ..rl_utilities.rl_agent_utils import ReplayBuffer, PrioritizedReplayBuffer
    from ..rl_envrionment_w_new_book_and_symm import (
        MarketMakingEnv,
        default_obs_fn,
        default_reward_fn,
        default_discrete_action_executor,
        DEFAULT_DISCRETE_ACTION_SPACE,
        DEFAULT_OBSERVATION_SPACE,
        NO_HAWKES_OBSERVATION_SPACE,
    )
except ImportError:
    libraries_dir = Path(__file__).resolve().parents[1]
    if str(libraries_dir) not in sys.path:
        sys.path.insert(0, str(libraries_dir))
    from rl_utilities.rl_agent_utils import ReplayBuffer, PrioritizedReplayBuffer
    from rl_envrionment_w_new_book_and_symm import (
        MarketMakingEnv,
        default_obs_fn,
        default_reward_fn,
        default_discrete_action_executor,
        DEFAULT_DISCRETE_ACTION_SPACE,
        DEFAULT_OBSERVATION_SPACE,
        NO_HAWKES_OBSERVATION_SPACE,
    )

"""
Dataclass with args for the agent and the env
"""
@dataclass
class Args:
    seed: int = 42069
    torch_deterministic: bool = True
    cuda: bool = True
    device: str = "cuda:0"
    save_model: bool = False
    run_dir: str = "NEW_runs"
    output_dir: str = "NEW_runs"
    reward_window: int = 100
    include_fees_in_reward: int = 1 # 1 for yes, 0 for no

    # Environment
    dt_per_step: float = 5/3600 # We let the agent act every 5 seconds
    inventory_penalty: float = 0.0
    terminal_inventory_penalty: float = 0.0
    discrete_quote_qty: int = 5
    max_inventory: int = 25
    include_hawkes_in_obs: bool = True


    end_ceiling: float = None
    ceiling_decrease_start_time: float = None

    # Algorithm
    total_timesteps: int = 3_000_000
    learning_rate: float = 5e-4
    buffer_size: int = 200_000
    gamma: float = 1.0
    tau: float = 1.0
    target_network_frequency: int = 20_000
    batch_size: int = 512
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.4
    learning_starts: int = 2_000
    train_frequency: int = 1
    log_frequency: int = 1000
    save_frequency: int = 50_000

    # Prioritized Experience Replay
    use_per: bool = False
    per_alpha: float = 0.6          # prioritization exponent (0 = uniform, 1 = full prioritization)
    per_beta_start: float = 0.4     # initial IS correction exponent
    per_beta_end: float = 1.0       # final IS correction (annealed linearly to ensure unbiased convergence)
    per_epsilon: float = 1e-6       # small constant added to TD errors to avoid zero priority
    per_mode: str = "proportional"  # "proportional" or "rank"

    # Dueling architecture
    use_dueling: bool = True
    # Number of hidden units in the Q-network (same for both regular and dueling)
    hidden_size: int = 64 
    # Number of hidden units in the value and advantage streams of the dueling network
    stream_hidden_size: int = 32 

    reward_fun_type: str = "asymm" # "asymm", "symm", or "pnl_only"
    pretrained_model_path: Optional[str] = None  # Path to a .pt file to initialize network weights from
    

"""
Function to make a folder that we can save stuff in for the training run
"""
def make_run_dir(args: Args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = args.output_dir if args.output_dir else args.run_dir
    run_dir = os.path.join(base_output_dir, f"DQN_mm_{timestamp}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

"""
JSON writer
"""
def write_json(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

"""
CSV writer
"""
def append_csv_row(path: str, fieldnames: list[str], row: dict):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


class CSVLogger:
    """Persistent CSV writer that keeps the file handle open for the run.
    Avoids the open/close/flush syscall overhead of re-opening on every row."""

    def __init__(self, path: str, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        file_exists = os.path.exists(path)
        self._handle = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=fieldnames)
        if not file_exists:
            self._writer.writeheader()
            self._handle.flush()

    def write(self, row: dict, flush: bool = False):
        self._writer.writerow(row)
        if flush:
            self._handle.flush()

    def close(self):
        try:
            self._handle.flush()
        finally:
            self._handle.close()

"""
The Q network, (regular and target)

Input layer size is determined by the dimension of the state space
The hidden layer has 256 nodes TODO optimize this
Output layer is the SIZE of action list
"""
class QNetwork(nn.Module):
    """
    Standard Q-network: shared trunk → single head producing Q(s,a) for all actions.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 64, stream_hidden_size: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.network(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling architecture (Wang et al. 2016):
    Shared feature extraction trunk splits into two streams:
      - Value stream:     V(s)    (scalar)
      - Advantage stream: A(s,a)  (per action)
    Aggregated as: Q(s,a) = V(s) + A(s,a) - mean_a'(A(s,a'))
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 64, stream_hidden_size: int = 32):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, stream_hidden_size),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(stream_hidden_size, stream_hidden_size),
            nn.ReLU(),
            nn.Linear(stream_hidden_size, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(stream_hidden_size, stream_hidden_size),
            nn.ReLU(),
            nn.Linear(stream_hidden_size, n_actions),
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)             # (batch, 1)
        advantage = self.advantage_stream(features)     # (batch, n_actions)
        # Subtract mean advantage for identifiability
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

"""
Schedule used to decay the exploration rate
"""
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

"""
Create an environment from the args class
"""
def make_env(args: Args) -> MarketMakingEnv:
    env_class = MarketMakingEnv
    
    env_kwargs = {
        "dt_per_step": args.dt_per_step,
        "inventory_penalty": args.inventory_penalty,
        "discrete_quote_qty": args.discrete_quote_qty,
        "max_inventory": args.max_inventory,
        "action_space": DEFAULT_DISCRETE_ACTION_SPACE,
        "action_executor": default_discrete_action_executor,
        "obs_fn": default_obs_fn,
        "observation_space": DEFAULT_OBSERVATION_SPACE if args.include_hawkes_in_obs else NO_HAWKES_OBSERVATION_SPACE,
        "reward_fn": default_reward_fn,
        "use_copula_for_non_improving": True,
        "use_copula_for_improving": True,
        "instrument_lifespan_hours": 16.5,
        "terminal_inventory_penalty": args.terminal_inventory_penalty,
        "include_fees_in_reward": args.include_fees_in_reward,
        "agent_seen_by_non_improving": False,
        "agent_seen_by_improving": True,
        "reward_fun_type": args.reward_fun_type,
        "end_ceiling": args.end_ceiling,
        "ceiling_decrease_start_time": args.ceiling_decrease_start_time,
        "include_hawkes_in_obs": args.include_hawkes_in_obs,
    }
    
    
    return env_class(**env_kwargs)

"""
Training function

"""
def train(args: Args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 
    torch.backends.cudnn.deterministic = args.torch_deterministic # Some CUDA optimizations are non-deterministic, so False can be used for reproducability

    device = torch.device(args.device)
    run_dir = make_run_dir(args)
    training_log_path = os.path.join(run_dir, "training_metrics.csv")
    args_path = os.path.join(run_dir, "args.json")
    latest_model_path = os.path.join(run_dir, "model_latest.pt")
    networks_dir = os.path.join(run_dir, "networks")
    os.makedirs(networks_dir, exist_ok=True)

    env = make_env(args)
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    write_json(
        args_path,
        {
            **vars(args),
            "device": str(device),
            "obs_dim": obs_dim,
            "n_actions": n_actions,
        },
    )

    # Select network architecture
    NetworkClass = DuelingQNetwork if args.use_dueling else QNetwork
    q_network = NetworkClass(obs_dim, n_actions, hidden_size=args.hidden_size, stream_hidden_size=args.stream_hidden_size).to(device) # Initialize the Q-network that we will use determine actions
    if args.pretrained_model_path is not None:
        state_dict = torch.load(args.pretrained_model_path, map_location=device)
        q_network.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {args.pretrained_model_path}")
    target_network = NetworkClass(obs_dim, n_actions, hidden_size=args.hidden_size, stream_hidden_size=args.stream_hidden_size).to(device) # Initialize the network we will use to compute TD-errors
    target_network.load_state_dict(q_network.state_dict()) # Initialize target network with same weights as Q-network
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate) # Standard, good optimizer that uses SGD

    # Select replay buffer
    if args.use_per:
        rb = PrioritizedReplayBuffer(
            args.buffer_size, env.observation_space.shape, device,
            alpha=args.per_alpha, epsilon=args.per_epsilon, mode=args.per_mode,
        )
    else:
        rb = ReplayBuffer(args.buffer_size, env.observation_space.shape, device)

    obs, _ = env.reset(seed=args.seed)
    # Pre-allocated GPU tensor for single-obs inference (avoids per-step alloc)
    _obs_tensor = torch.empty((1, obs_dim), dtype=torch.float32, device=device)
    episode_return = 0.0
    episode_len = 0
    episode_count = 0
    episode_returns: deque[float] = deque(maxlen=args.reward_window)
    window_sum = 0.0
    last_episode_return = 0.0
    last_loss = None
    last_mean_q = None
    start_time = time.time()
    reward_since_last_log = 0.0
    steps_since_last_log = 0

    training_csv = CSVLogger(
        training_log_path,
        [
            "global_step",
            "elapsed_sec",
            "epsilon",
            "loss",
            "mean_q",
            "avg_reward_since_last_log",
            "avg_return_window",
            "last_episode_return",
            "episodes_completed",
            "replay_size",
            "sps",
        ],
    )

    print(f"Logging run artifacts to {run_dir}")


    # Through the episodes
    for step in range(args.total_timesteps):
        if step % 1000 == 0 and step > 0:
            print(f'At episode {episode_count}, step {step}/{args.total_timesteps}...')
            
        epsilon = linear_schedule(args.start_e, args.end_e, int(args.exploration_fraction * args.total_timesteps), step)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Random action (exploration)
            action = int(env.action_space.sample())
        else:
            with torch.no_grad():
                # Exploitation, select the best action as per the Q-network
                _obs_tensor[0].copy_(torch.from_numpy(obs))
                q_vals = q_network(_obs_tensor)
                action = int(q_vals.argmax(dim=1).item())

        # Take the action and observe the next state and reward
        next_obs, reward, terminated, truncated, info = env.step(action)
        # Check termination
        done = terminated or truncated
        episode_return += reward
        episode_len += 1
        reward_since_last_log += reward
        steps_since_last_log += 1
        
        # Add the experience to the replay buffer
        rb.add(obs, next_obs, action, reward, done)
        obs = next_obs

        if done:
            episode_count += 1
            ep_ret_f = float(episode_return)
            if len(episode_returns) == episode_returns.maxlen:
                window_sum -= episode_returns[0]
            episode_returns.append(ep_ret_f)
            window_sum += ep_ret_f
            last_episode_return = ep_ret_f

            obs, _ = env.reset()
            episode_return = 0.0
            episode_len = 0

        
        # If we don't want to learn until we have some experience, we can skip the learning part until we have at least args.learning_starts experience in the replay buffer
        if step < args.learning_starts:
            continue
        
        # Train the Q-network
        if step % args.train_frequency == 0:

            """
            s_obs: batch of states (observations)
            s_next: batch of next states
            s_act: batch of actions taken
            s_rew: batch of rewards received
            s_done: batch of done flags indicating episode termination
            """

            # Sample from replay buffer (PER returns extra indices + IS weights)
            if args.use_per:
                beta = args.per_beta_start + (args.per_beta_end - args.per_beta_start) * (step / args.total_timesteps)
                s_obs, s_next, s_act, s_rew, s_done, per_indices, is_weights = rb.sample(args.batch_size, beta=beta)
            else:
                s_obs, s_next, s_act, s_rew, s_done = rb.sample(args.batch_size)

            with torch.no_grad():
                """
                Now that we have sampled a minibatch:
                We compute the td target for each experience in the batch using target network
                target_max is the best action q-value according to target network (for the next state)
                """
                best_actions = q_network(s_next).argmax(dim=1)
                target_max = target_network(s_next).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                td_target = s_rew + args.gamma * target_max * (1.0 - s_done)

            q_vals = q_network(s_obs).gather(1, s_act.unsqueeze(1)).squeeze(1)

            # Per-element Huber loss, weighted by IS weights if using PER
            elementwise_loss = F.smooth_l1_loss(q_vals, td_target, reduction='none')
            if args.use_per:
                loss = (is_weights * elementwise_loss).mean()
            else:
                loss = elementwise_loss.mean()
            
            last_loss = float(loss.item())
            last_mean_q = float(q_vals.mean().item())

            # Update the Q-network by minimizing the loss
            optimizer.zero_grad() # Zero out the gradients before backpropagation
            loss.backward() # Backpropagation to compute gradients
            optimizer.step() # Update the weights of the Q-network using the optimizer

            # Update priorities in the replay buffer with fresh TD errors
            if args.use_per:
                td_errors = (td_target - q_vals).detach().abs().cpu().numpy()
                rb.update_priorities(per_indices, td_errors)

            # Log training metrics every args.log_frequency steps, including:
            # - Loss
            # - Mean Q-value of the batch
            if step % args.log_frequency == 0:
                sps = int(step / max(time.time() - start_time, 1e-9))
                elapsed_sec = time.time() - start_time
                avg_return_window = (window_sum / len(episode_returns)) if episode_returns else 0.0
                avg_rew_log = reward_since_last_log / max(steps_since_last_log, 1)

                training_csv.write(
                    {
                        "global_step": step,
                        "elapsed_sec": elapsed_sec,
                        "epsilon": float(epsilon),
                        "loss": last_loss,
                        "mean_q": last_mean_q,
                        "avg_reward_since_last_log": avg_rew_log,
                        "avg_return_window": avg_return_window,
                        "last_episode_return": last_episode_return,
                        "episodes_completed": episode_count,
                        "replay_size": rb.size,
                        "sps": sps,
                    },
                    flush=True,
                )

                print(
                    f"  step={step}  loss={last_loss:.4f}  q={last_mean_q:.4f}  "
                    f"eps={epsilon:.3f}  avg_rew={avg_rew_log:.4f}  avg{args.reward_window}={avg_return_window:.2f}  SPS={sps}"
                )

                reward_since_last_log = 0.0
                steps_since_last_log = 0

        """
        Update the target network every once in a while
        Tau determines how much of the change in the weights of the Q-network we copy to target network, tau=1 means we set target network weights equal to Q-network's
        We don't do it every time to stabilize training, but we also don't want to do it too rarely to avoid the target network becoming too stale
        """
        if step % args.target_network_frequency == 0:
            if args.tau == 1.0:
                target_network.load_state_dict(q_network.state_dict())
            else:
                for tp, qp in zip(target_network.parameters(), q_network.parameters()):
                    tp.data.copy_(args.tau * qp.data + (1.0 - args.tau) * tp.data)

        if args.save_frequency > 0 and step % args.save_frequency == 0 and step > 0:
            checkpoint_path = os.path.join(networks_dir, f"model_step{step}.pt")
            torch.save(q_network.state_dict(), checkpoint_path)

    torch.save(q_network.state_dict(), latest_model_path)
    print(f"Model saved to {latest_model_path}")

    write_json(
        os.path.join(run_dir, "summary.json"),
        {
            "episodes_completed": episode_count,
            "total_timesteps": args.total_timesteps,
            "elapsed_sec": time.time() - start_time,
            "final_avg_return_window": (window_sum / len(episode_returns)) if episode_returns else 0.0,
            "last_episode_return": last_episode_return,
            "last_loss": last_loss,
            "last_mean_q": last_mean_q,
        },
    )

    training_csv.close()
    env.close()
    return q_network


if __name__ == "__main__":

    for learning_rate in [2.5e-3]: #128, 256, 
        for seeder in [696969]: 
            for dt in [5]: #696969 , , 420420
                for terminal_inventory_penalty in [0]: #0.1, 1.0, 5.0
                    
                    args = Args(
                    dt_per_step=dt/3600,
                    total_timesteps=2_000_000,
                    learning_starts=2_000,
                    use_dueling=False,
                    seed=seeder,
                    device="cuda:1",
                    hidden_size=64,
                    stream_hidden_size=64,
                    learning_rate=learning_rate,
                    target_network_frequency=20_000,
                    run_dir="SYMMETRIC_MODELS/Hyperparam_Sweep",
                    output_dir="SYMMETRIC_MODELS/Hyperparam_Sweep",
                    use_per=False,
                    batch_size=256,
                    
                    reward_fun_type='asymm',
                    inventory_penalty=0.001,
                    terminal_inventory_penalty=terminal_inventory_penalty,

                    end_ceiling=None,
                    ceiling_decrease_start_time=None,
                    end_e=0.1,
                    exploration_fraction=1.0,
                    include_hawkes_in_obs=False,
                  
                    )
                    train(args)

    # lr_list = [1e-4, 5e-4]
    # inventory_penalty_list = [0.01, 0.1]
    # use_per_list = [True, False]
    # #gamma_list = [0.95, 0.99]
    # hidden_hiddenstream_list = [(64, 32), (32, 16)]

    # # Run a sweep of the hyperparams
    # for lr in lr_list:
    #     for inventory_penalty in inventory_penalty_list:
    #         for use_per in use_per_list:
    #             for hidden_hiddenstream in hidden_hiddenstream_list:
    #                 print(f"Running with lr={lr}, inventory_penalty={inventory_penalty}, use_per={use_per}, hidden_hiddenstream={hidden_hiddenstream}")
    #                 # Create a new Args object with the current hyperparams
    #                 for seeder in [67, 69]:
    #                     args = Args(
    #                         learning_rate=lr,
    #                         inventory_penalty=inventory_penalty,
    #                         use_per=use_per,
    #                         hidden_size=hidden_hiddenstream[0],
    #                         stream_hidden_size=hidden_hiddenstream[1],
    #                         total_timesteps=int(28/28.5*16*342*3600/8), # equals 28 days of training
    #                         output_dir="Hyperparam_Sweep_Results",
    #                         end_e=0.905,
    #                         exploration_fraction=1.0,
    #                         seed=seeder,
    #                     )
                        
    #                     train(args)


            
            

    


