""""
The PPO logic is based on the Higgsfield GitHub repo in the following link: https://github.com/higgsfield-ai/higgsfield/blob/main/higgsfield/rl/rl\_adventure\_2/3.ppo.ipynb
"""

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
from torch.distributions import Categorical, Normal, Independent

try:
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
    reward_window: int = 20
    include_fees_in_reward: int = 0 # 1 for yes, 0 for no

    # Environment
    dt_per_step: float = 1/3600 # We let the agent act every 1 second
    inventory_penalty: float = 0.0
    terminal_inventory_penalty: float = 0.0
    discrete_quote_qty: int = 5
    max_inventory: int = 25

    # Algorithm
    total_timesteps: int = int(3_000_000) # 8 hours of training, roughly
    learning_rate: float = 5e-4
    gamma: float = 1
    tau: float = 0.95          # GAE lambda (not a target-network tau). 1.0 == pure MC returns over the
                               # rollout (very high variance); 0.95 is the standard PPO setting and
                               # cuts critic-target variance a lot like DQN's bootstrapped TD targets.
    entropy_weight: float = 0.01
    learning_starts: int = 2000
    train_frequency: int = 4096
    mini_batch_size: int = 128
    ppo_epochs: int = 8
    clip_param: float = 0.2
    # Stability knobs
    max_grad_norm: float = 0.5          # global gradient-norm clip (PPO standard)
    value_loss_coef: float = 0.5        # weight on critic MSE in the total loss
    reward_clip: float = 10.0           # clip the normalised reward to +/- this value
    log_frequency: int = 4096
    save_frequency: int = 50_000

    # Number of hidden units in the neural network
    hidden_size_1: int = 256
    hidden_size_2: int = 256
    critic_size: int = 256

    reward_fun_type: str = "pnl_only" # "asymm", "symm", or "pnl_only"
    include_hawkes_in_obs: bool = True
    pretrained_model_path: Optional[str] = None  # Path to a .pt file to initialize network weights from
    

"""
Function to make a folder that we can save stuff in for the training run
"""
def make_run_dir(args: Args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = args.output_dir if args.output_dir else args.run_dir
    run_dir = os.path.join(base_output_dir, f"PPO_mm_{timestamp}_seed{args.seed}")
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
The Actor critic network

Input layer size is determined by the dimension of the state space
The hidden layer has 256 nodes TODO optimize this
Output layer is the SIZE of action list
"""


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size_1, hidden_size_2, critic_size):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, critic_size),
            nn.ReLU(),
            nn.Linear(critic_size, critic_size),
            nn.ReLU(),
            nn.Linear(critic_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, num_outputs)
        )
        
    # UPDATED
    def forward(self, x):
        value = self.critic(x)
        logits = self.actor(x)
        dist = Categorical(logits = logits)
        return dist, value


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
        "include_hawkes_in_obs": args.include_hawkes_in_obs,
    }
    
    
    return env_class(**env_kwargs)


"Function for computing GAE"
# PAGE 5 IN OPENAI ARTICLE
def compute_gae(next_value, rewards, masks, values, gamma, tau):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

"Functions for selecting actions from the state space"
def select_action(model, state):
    discrete_dist, value = model(state)
    action = discrete_dist.sample()                      # shape: (1,)

    # Joint log-prob = discrete log-prob
    log_prob = discrete_dist.log_prob(action)

    return action, log_prob, value


"PPO functions for iteration and updates"
def ppo_iter(mini_batch_size, states, discr_action, log_probs, returns, advantage):
    """Yield disjoint minibatches covering the full rollout exactly once.

    The previous implementation used ``np.random.randint`` (sampling **with
    replacement**) and only produced ``batch_size // mini_batch_size``
    minibatches per epoch, so some transitions were never used and others
    were used many times. Standard PPO permutes the batch and slices it.
    """
    batch_size = states.size(0)
    indices = np.random.permutation(batch_size)
    for start in range(0, batch_size - mini_batch_size + 1, mini_batch_size):
        rand_ids = indices[start:start + mini_batch_size]
        yield states[rand_ids], discr_action[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids]


def ppo_update(ppo_epochs, mini_batch_size, states, discr_actions, log_probs, returns, advantages, optimizer, model, entropy_weight, clip_param, max_grad_norm, value_loss_coef):
    for _ in range(ppo_epochs):
        for state, discr_action, old_log_probs, return_, advantage_ in ppo_iter(mini_batch_size, states, discr_actions, log_probs, returns, advantages):
            discrete_dist, value = model(state)
            # Total entropy = discrete entropy averaged over the mini-batch.
            entropy = (
                discrete_dist.entropy()
            ).mean()
            new_log_probs = (
                discrete_dist.log_prob(discr_action)
            )
            # Ratio gives the r(theta)
            ratio = (new_log_probs - old_log_probs).exp()

            # Gives the surrogate function, note that this is where we limit the size of changes
            surr1 = ratio * advantage_
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage_

            # The actor gets the loss specified in the article
            actor_loss  = - torch.min(surr1, surr2).mean()
            # Critic: Huber (smooth L1) loss 
            critic_loss = F.smooth_l1_loss(value.squeeze(-1), return_)

            loss = value_loss_coef * critic_loss + actor_loss - entropy_weight * entropy

            optimizer.zero_grad()
            loss.backward()
            # Global gradient-norm clip
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

    return loss.item(), value.mean().item()



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
    num_inputs  = env.observation_space.shape[0]
    num_discrete = env.action_space.n

    write_json(
        args_path,
        {
            **vars(args),
            "device": str(device),
            "obs_dim": int(num_inputs),
            "num_discrete": int(num_discrete),
        },
    )

    # Select network architecture
    model = ActorCritic(num_inputs, num_discrete, args.hidden_size_1, args.hidden_size_2, args.critic_size).to(device)
    if args.pretrained_model_path is not None:
        state_dict = torch.load(args.pretrained_model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {args.pretrained_model_path}")
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) # Standard, good optimizer that uses SGD

    obs, _ = env.reset(seed=args.seed)
    # Pre-allocated GPU tensor for single-obs inference (avoids per-step alloc)
    _obs_tensor = torch.empty((1, num_inputs), dtype=torch.float32, device=device)

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
            "entropy weight",
            "loss",
            "mean_q",
            "avg_reward_since_last_log",
            "avg_return_window",
            "last_episode_return",
            "episodes_completed",
            "sps",
        ],
    )

    print(f"Logging run artifacts to {run_dir}")

    log_probs = []
    values    = []
    states    = []
    discr_actions = []
    rewards   = []
    masks     = []    
    # Through the episodes
    for step in range(args.total_timesteps):
        if step % 1000 == 0 and step > 0:
            print(f'At episode {episode_count}, step {step}/{args.total_timesteps}...')
            
        entropy_weight = args.entropy_weight

        # Action selection:
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        discr_action, log_prob, value = select_action(model, state_t)
        action = int(discr_action.item())

        # Take the action and observe the next state and reward
        next_obs, reward, terminated, truncated, info = env.step(action)

        reward = float(reward)

        # Save  relevant info
        log_probs.append(log_prob)
        values.append(value.squeeze(-1))
        rewards.append(torch.FloatTensor([reward]).to(device))
        masks.append(torch.FloatTensor([1.0 - float(terminated)]).to(device))
        states.append(state_t)
        discr_actions.append(discr_action)

        # Check termination
        done = terminated or truncated
        episode_return += reward
        episode_len += 1
        reward_since_last_log += reward
        steps_since_last_log += 1

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
            # estimator doesn't mix returns from different episodes.

        
        # If we don't want to learn until we have some experience, we can skip the learning part until we have at least args.learning_starts experience in the replay buffer
        if step < args.learning_starts:
            continue
        
        # Train the PPO-network
        if step % args.train_frequency == 0:

            """
            s_obs: batch of states (observations)
            s_next: batch of next states
            s_act: batch of actions taken
            s_rew: batch of rewards received
            s_done: batch of done flags indicating episode termination
            """
            # Update the neural network
            next_state_t = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
            with torch.no_grad():
                _, next_value = model(next_state_t)
            next_value = next_value.squeeze(-1)

            # Compute advantage vector
            returns = compute_gae(next_value, rewards, masks, values, args.gamma, args.tau)

            # Aggregate all the info we saved
            returns       = torch.cat(returns).detach()
            log_probs_t   = torch.cat(log_probs).detach()
            values_t      = torch.cat(values).detach()
            states_t      = torch.cat(states).detach()
            discr_acts_t  = torch.cat(discr_actions).detach()
            advantage     = returns - values_t
            # Standard PPO trick: normalize advantages for stability
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) 

            # Update the PPO network using the function defined above
            last_loss, last_mean_q = ppo_update(
                args.ppo_epochs, args.mini_batch_size,
                states_t, discr_acts_t,
                log_probs_t, returns, advantage, optimizer, model, entropy_weight, args.clip_param,
                args.max_grad_norm, args.value_loss_coef,
            )

            log_probs.clear()
            values.clear() 
            states.clear()
            discr_actions.clear()
            rewards.clear()
            masks.clear()

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
                        "entropy weight": float(entropy_weight),
                        "loss": last_loss,
                        "mean_q": last_mean_q,
                        "avg_reward_since_last_log": avg_rew_log,
                        "avg_return_window": avg_return_window,
                        "last_episode_return": last_episode_return,
                        "episodes_completed": episode_count,
                        "sps": sps,
                    },
                    flush=True,
                )

                print(
                    f"  step={step}  loss={last_loss:.4f}  v={last_mean_q:.4f}  "
                    f"  avg_rew={avg_rew_log:.4f}  last_ep_ret={last_episode_return:.2f}  avg_ret_window={avg_return_window:.2f}  SPS={sps}  Hours left={(args.total_timesteps - step)/sps/3600:.2f}"
                )

                reward_since_last_log = 0.0
                steps_since_last_log = 0

        """
        Update the target network every once in a while
        Tau determines how much of the change in the weights of the Q-network we copy to target network, tau=1 means we set target network weights equal to Q-network's
        We don't do it every time to stabilize training, but we also don't want to do it too rarely to avoid the target network becoming too stale
        """

        if args.save_frequency > 0 and step % args.save_frequency == 0 and step > 0:
            checkpoint_path = os.path.join(networks_dir, f"model_step{step}.pt")
            torch.save(model.state_dict(), checkpoint_path)

    torch.save(model.state_dict(), latest_model_path)
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
    return model


if __name__ == "__main__":
    # for value_loss_coef in [0.5, 0.75, 1]:
    #         for learning_rate in [5e-4, 2.5e-3]:
    #                 for entropy_weight in [0.01, 0.005]:
    #                     print(f"Running with value_loss_coef={value_loss_coef}, learning_rate={learning_rate}, entropy_weight={entropy_weight}")  
    #                     args = Args(
    #                         seed=666777,
    #                         device="cuda:0",
    #                         value_loss_coef =value_loss_coef,
    #                         learning_rate=learning_rate,
    #                         entropy_weight= entropy_weight,
    #                         run_dir="HYPERSWEEP_PPO",
    #                         output_dir="HYPERSWEEP_PPO",
    #                     )
    #                     train(args)

    # for clipping_range in [0.1, 0.2, 0.3]:
    #         for epochs in [4, 10, 15]:
    #                 print(f"Running with clipping_range={clipping_range}, epochs={epochs}")  
    #                 args = Args(
    #                     seed=666777,
    #                     device="cpu",
    #                     clip_param=clipping_range,
    #                     ppo_epochs=epochs,
    #                     run_dir="HYPERSWEEP_PPO",
    #                     output_dir="HYPERSWEEP_PPO",
    #                 )
    #                 train(args)

    # for entropy_weight in [0.005, 0.01]:
    #                 print(f"entropy weight {entropy_weight}")  
    #                 args = Args(
    #                     seed=666777,
    #                     device="cpu",
    #                     entropy_weight=entropy_weight,
    #                     total_timesteps=14_850_000,
    #                     run_dir="HYPERSWEEP_PPO",
    #                     output_dir="HYPERSWEEP_PPO",
    #                 )
    #                 train(args)


    seeds = [676767, 420420, 696969]

    for seed_ in [676767, 696969]:
        for dt in [5]:
            for reward_fun_type in ['symm']:
                for tau_ in [0.9,0.95,1]:
                    print(f"Running with seed={seed_}, dt={dt}, reward_fun_type={reward_fun_type}, tau={tau_}")
                    args = Args(
                        tau=tau_,
                        seed=seed_,
                        device="cpu",
                        inventory_penalty=0.001,
                        reward_fun_type=reward_fun_type,
                        dt_per_step=dt/3600,
                        total_timesteps=2_000_000,
                        run_dir="SYMMETRIC_MODELS/PPOS/HYPER",
                        output_dir="SYMMETRIC_MODELS/PPOS/HYPER",
                        include_hawkes_in_obs=False,
                        include_fees_in_reward=1,
                       # pretrained_model_path="ppo_symm_pretrained.pt",
                    )
                    train(args)

    