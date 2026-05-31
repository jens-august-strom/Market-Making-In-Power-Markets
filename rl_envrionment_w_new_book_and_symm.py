"""
This file is used to test an alternative version of the rl envrionment where we symmetrize the parameters for bids and asks:
LO arrival will use bid arrival parameters for both sides
Both for improving and non-improving
Same for MO's
We also use same parameters for bid/ask for all other aspects
"""

from __future__ import annotations

from dataclasses import dataclass
import bisect
import importlib
import importlib.util
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from scipy.stats import beta
from scipy.stats import t as student_t
import numpy as np


if importlib.util.find_spec("gymnasium") is not None:
    gym = importlib.import_module("gymnasium")
    spaces = gym.spaces
print("importing the OB")
try:
    from .orderbook import Book, Fill
    from .hawkes_fit_library import sim_hawkes
except ImportError:
    from orderbook import Book, Fill
    from hawkes_fit_library import sim_hawkes

print("rl_environment was reloaded")

# ── Module-level copula array cache (shared across all env instances) ──────────
_COPULA_CACHE: Dict[str, Any] = {}


def _load_copula_arrays(path: str | Path) -> Dict[str, Any]:
    """Load and cache copula npz arrays by path. Shared (read-only) across all env instances."""
    key = str(_resolve_resource_path(path))
    if key not in _COPULA_CACHE:
        data = np.load(_resolve_resource_path(path))
        n = int(data['n'])
        _COPULA_CACHE[key] = {
            'sorted_time_raw':    data['sorted_time_raw'],
            'ranks_delta_sorted': data['ranks_delta_sorted'],
            'ranks_qty_sorted':   data['ranks_qty_sorted'],
            'sorted_delta':       data['sorted_delta'],
            'sorted_qty':         data['sorted_qty'],
            'log_norm':           data['log_norm'],
            # r_array[i] = i + 1, so r_array[lo:hi] gives ranks lo+1..hi (zero-copy view)
            'r_array':            np.arange(1, n + 1, dtype=np.float64),
            'n':                  n,
            'W':                  int(data['W']),
        }
    elif 'r_array' not in _COPULA_CACHE[key]:
        n = _COPULA_CACHE[key]['n']
        _COPULA_CACHE[key]['r_array'] = np.arange(1, n + 1, dtype=np.float64)
    return _COPULA_CACHE[key]


# ── Improving-LO copula cache (separate; different npz field names) ────────────
_IMPROVING_COPULA_CACHE: Dict[str, Any] = {}


def _load_improving_copula_arrays(path: str | Path) -> Dict[str, Any]:
    """Load and cache improving-LO copula arrays.

    Fields differ from the non-improving copula: uses ``ranks_rpp_sorted``
    (relative-price-position ranks) instead of ``ranks_delta_sorted`` /
    ``sorted_delta``, because the price marginal is inverted parametrically
    via Beta.ppf rather than via empirical quantile.
    """
    key = str(_resolve_resource_path(path))
    if key not in _IMPROVING_COPULA_CACHE:
        data = np.load(_resolve_resource_path(path))
        n = int(data['n'])
        _IMPROVING_COPULA_CACHE[key] = {
            'sorted_time_raw':  data['sorted_time_raw'],
            'ranks_rpp_sorted': data['ranks_rpp_sorted'],
            'ranks_qty_sorted': data['ranks_qty_sorted'],
            'sorted_qty':       data['sorted_qty'],
            'log_norm':         data['log_norm'],
            'r_array':          np.arange(1, n + 1, dtype=np.float64),
            'n':                n,
            'W':                int(data['W']),
        }
    elif 'r_array' not in _IMPROVING_COPULA_CACHE[key]:
        n = _IMPROVING_COPULA_CACHE[key]['n']
        _IMPROVING_COPULA_CACHE[key]['r_array'] = np.arange(1, n + 1, dtype=np.float64)
    return _IMPROVING_COPULA_CACHE[key]


@dataclass
class BackgroundEvent:
    time: float
    event_type: str


def _safe_float(value: Optional[float], default: float = 0.0) -> float:
    return float(default if value is None else value)


def _resolve_resource_path(path: str | Path) -> Path:
    resource_path = Path(path)
    if resource_path.is_absolute():
        return resource_path
    return (Path(__file__).resolve().parent / resource_path).resolve()


IMPROVING_LO_QTY_PARAMS_GE_05 = {
    "atoms": np.array([5.0, 1.0, 2.0], dtype=float),
    "atom_probs": np.array([0.274898, 0.139298, 0.129856], dtype=float),
    "p_rem": 0.455947,
    "shape": 1.499100,
    "scale": 1.404925,
}


IMPROVING_LO_QTY_PARAMS_LT_05 = {
    "atoms": np.array([5.0, 4.0], dtype=float),
    "atom_probs": np.array([0.425870, 0.104026], dtype=float),
    "p_rem": 0.470105,
    "shape": 0.756465,
    "scale": 4.475185,
}

"""
Function to get the mid price

If there is a valid mid price, return it and update the last mid price
If there is no valid mid price, return the last mid price if it exists
If there is no last mid price, try to estimate it from the best bid and ask

If there is no valid mid price and no best bid or ask, return 0.0
"""
def _mark_price(env: "MarketMakingEnv") -> float:
    mid = env.book.mid_price()
    if mid is not None:
        env._last_mid = float(mid)
        return float(mid)

    if env._last_mid is not None:
        return float(env._last_mid)

    bb = env.book.best_bid()
    ba = env.book.best_ask()
    if bb is not None and ba is not None:
        env._last_mid = 0.5 * (bb + ba)
    elif bb is not None:
        env._last_mid = float(bb)
    elif ba is not None:
        env._last_mid = float(ba)
    else:
        env._last_mid = 0.0
    return float(env._last_mid)

"""
Default observation (State) function:
- Mid price
- Spread
- Best bid
- Best ask
- Book pressure on bids (depth 3)
- Book pressure on asks (depth 3)
- Inventory (normalized by max_inventory)
- Cash
- Time remaining (normalized by total time)
"""
def default_obs_fn(env: "MarketMakingEnv") -> np.ndarray:
    book = env.book
    # Top-3 anonymous levels per side (price + qty each); missing levels filled with 0.0
    bid_levels = book.top_n_bid_anon(3)  # list of up to 3 (price, anon_qty), best bid first
    ask_levels = book.top_n_ask_anon(3)  # list of up to 3 (price, anon_qty), best ask first

    _cur_ceil = env.current_max_inventory()
    max_inv = _cur_ceil if _cur_ceil > 0 else 1
    end_time = book.end_time
    time_remaining = end_time - book.time
    if time_remaining < 0.0:
        time_remaining = 0.0
    time_norm = time_remaining / end_time if end_time > 1e-9 else 0.0

    # Compute mid price once; used for level offsets and mid-price delta
    current_mark_price = _mark_price(env)

    include_hawkes = getattr(env, "include_hawkes_in_obs", True)
    obs_size = 31 if include_hawkes else 15
    obs = np.empty(obs_size, dtype=np.float32)

    obs[0] = env.inventory / max_inv

    # Bid levels L1, L2, L3 — (offset from mid, anon_qty) interleaved; 0.0 if level absent
    # offset = mid - bid_price  (positive = below mid, increases with depth)
    for i in range(3):
        if i < len(bid_levels):
            obs[1 + i * 2] = current_mark_price - bid_levels[i][0]
            obs[2 + i * 2] = bid_levels[i][1]
        else:
            obs[1 + i * 2] = 0.0
            obs[2 + i * 2] = 0.0

    # Ask levels L1, L2, L3 — (offset from mid, anon_qty) interleaved; 0.0 if level absent
    # offset = ask_price - mid  (positive = above mid, increases with depth)
    for i in range(3):
        if i < len(ask_levels):
            obs[7 + i * 2] = ask_levels[i][0] - current_mark_price
            obs[8 + i * 2] = ask_levels[i][1]
        else:
            obs[7 + i * 2] = 0.0
            obs[8 + i * 2] = 0.0

    obs[13] = time_norm

    if include_hawkes:
        # We will give it all of the active excitation kernels
        exc = env.exc_kernel_matrix
        last_decay = env.last_decay_time

        if last_decay < book.time:
            # Apply decay to the excitation kernels before returning the observation
            exc *= np.exp(-env.beta_matrix * (book.time - last_decay))
            env.last_decay_time = book.time

        # HAWKES EXCITATION KERNELS =============================================
        obs[14] = exc[0, 0]
        obs[15] = exc[0, 5]

        obs[16] = exc[1, 0]
        obs[17] = exc[1, 1]
        obs[18] = exc[1, 2]
        obs[19] = exc[1, 4]
        obs[20] = exc[1, 5]

        obs[21] = exc[2, 2]
        obs[22] = exc[2, 4]

        obs[23] = exc[3, 0]
        obs[25] = exc[3, 2]
        obs[24] = exc[3, 3]
        obs[26] = exc[3, 4]
        obs[27] = exc[3, 5]

        obs[28] = exc[4, 4]

        obs[29] = exc[5, 5]

        # Mid-price delta (index 30 with Hawkes)
        previous_mark_price = env.prev_mark_price
        if previous_mark_price is None:
            obs[30] = 0.0
        else:
            obs[30] = current_mark_price - previous_mark_price
    else:
        # Mid-price delta (index 14 without Hawkes)
        previous_mark_price = env.prev_mark_price
        if previous_mark_price is None:
            obs[14] = 0.0
        else:
            obs[14] = current_mark_price - previous_mark_price

    env.prev_mark_price = current_mark_price

    return obs

def calc_fees(env: "MarketMakingEnv", fills: List[Fill]) -> float:
    exchange_and_clearing_rate = 0.09 + 0.035
    bank_rate = 0.0025
    bank_min_per_trade = 0.01

    step_fees = 0.0

    for fill in fills:
        qty = max(0.0, float(fill.qty))
        if qty <= 0.0:
            continue
        bank_fee = max(bank_rate * qty, bank_min_per_trade)
        step_fees += exchange_and_clearing_rate * qty + bank_fee

    for _side, _price, qty_raw in env._taker_trades:
        qty = max(0.0, float(qty_raw))
        if qty <= 0.0:
            continue
        bank_fee = max(bank_rate * qty, bank_min_per_trade)
        step_fees += exchange_and_clearing_rate * qty + bank_fee

    return step_fees

"""
Default reward function:
Change in PnL minus penalty for inventory risk (quadratic penalty on inventory)

Fees:
To EPEX: 0.09 per MWh traded
To clearing house: 0.035 per MWh traded
To bank: 0.0025 per MWh traded
Bank minimum fee per trade: 0.01
"""
def default_reward_fn(
    env: "MarketMakingEnv",
    action: np.ndarray,
    fills: List[Fill],
    info: Dict[str, Any],
) -> float:
    del action, info

    # Total fee per trade:
    # - EPEX: 0.09 per MWh
    # - Clearing house: 0.035 per MWh
    # - Bank: 0.0025 per MWh with a minimum of 0.01 per trade
    exchange_and_clearing_rate = 0.09 + 0.035
    bank_rate = 0.0025
    bank_min_per_trade = 0.01

    step_fees = calc_fees(env, fills)

    mp = _mark_price(env)
    env.last_observed_mid = mp
    env.step_fees = step_fees
    pnl = env.cash + env.inventory * mp
    asym_spec_penalty = (mp - env.prev_mark_price) * env._prev_inventory
    reward = (pnl - env._prev_pnl) - env.inventory_penalty * float(env.inventory ** 2) - step_fees*env.include_fees_in_reward
    
    if env.reward_fun_type == "asymm":
        reward = reward - max(asym_spec_penalty, 0.0)
    elif env.reward_fun_type == "symm":
        reward = reward - asym_spec_penalty
    
    env.prev_mark_price = mp
    env._prev_pnl = pnl
    env._prev_inventory = env.inventory
    return float(reward)

"""
Default action executer:
Execute the action by placing limit orders at the specified offsets from the mid price with the specified quantities.
Action format: [bid_offset, ask_offset, bid_qty, ask_qty]

Will automatically cancel existing agent orders if auto_cancel is True (handled in step function).

TODO: Add action for clearing inventory
      Add agent order to hawkes history should be updated to the new keys
"""
def default_action_executor(env: "MarketMakingEnv", action: np.ndarray) -> None:
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_arr.shape[0] != 4:
        raise ValueError("action must have 4 values: [bid_offset, ask_offset, bid_qty, ask_qty]")

    bid_offset, ask_offset, bid_qty_raw, ask_qty_raw = [float(x) for x in action_arr]
    bid_qty = max(0.0, round(bid_qty_raw, 1))
    ask_qty = max(0.0, round(ask_qty_raw, 1))

    mark = _mark_price(env)
    bid_price = max(0.0, mark - bid_offset)
    ask_price = max(0.0, mark + ask_offset)

    if bid_qty > 0:
        _place_agent_limit_order(env, "B", bid_price, bid_qty)

    if ask_qty > 0:
        _place_agent_limit_order(env, "S", ask_price, ask_qty)


QUOTE_STYLES = ["best", "level", "none"] #  "deep_1", "deep_2", "deep_3"
QUANTITY_CHOICES = [1.0]

QUOTE_ACTIONS = [(s, q) for s in QUOTE_STYLES for q in QUANTITY_CHOICES]
N_QUOTE = len(QUOTE_ACTIONS)  # 3

# Action layout:
#   0 .. N_QUOTE*N_QUOTE-1 : bid_style x ask_style grid (9 combos)
#                            "none" style on a side means that side does not quote.
#                            "none x none" (action 8) = both sides idle.
#   N_QUOTE*N_QUOTE         : dump inventory  [9]
NO_QUOTE_ACTION_ID = QUOTE_STYLES.index("none") * N_QUOTE + QUOTE_STYLES.index("none")  # 8
DUMP_ACTION_ID     = N_QUOTE * N_QUOTE                                                   # 9

DEFAULT_DISCRETE_ACTION_SPACE = spaces.Discrete(DUMP_ACTION_ID + 1)  # 10


def _normalize_discrete_action(action: Any) -> int:
    action_arr = np.asarray(action)
    if action_arr.shape == ():
        return int(action_arr)
    if action_arr.size == 1:
        return int(action_arr.reshape(-1)[0])
    raise ValueError(f"discrete action must be a scalar integer in [0, {DUMP_ACTION_ID}]")


def _agent_lo_hawkes_key(env: "MarketMakingEnv", side: str) -> str:
    side_up = str(side).upper()
    if env.include_LO_improve:
        return "LO_bid_no_improve" if side_up == "B" else "LO_ask_no_improve"
    return "LO_bid" if side_up == "B" else "LO_ask"


def _place_agent_limit_order(env: "MarketMakingEnv", side: str, price: float, qty: float) -> None:
    if qty <= 0:
        return

    order_id, fills = env.book.add_limit(str(side).upper(), float(price), float(qty), env.book.time, agent=True)
    env._new_agent_order_ids.append(order_id)
    # NOTE: Do NOT inject agent orders into the Hawkes history.
    # The background Hawkes process is calibrated on historical data and must
    # stay exogenous.  With dt_per_step=1/3600 and a 0.02 h history window
    # the agent would add ~144 artificial events vs ~5-10 natural ones,
    # inflating intensity 10-25x and collapsing the spread via excess
    # improving LOs.
    for fill in fills:
        env._taker_trades.append((str(side).upper(), float(fill.price), float(fill.qty)))


def _reference_quotes(env: "MarketMakingEnv", tick_size: float = 0.01) -> Tuple[float, float]:
    best_bid = env.book.best_bid()
    best_ask = env.book.best_ask()

    if best_bid is None:
        best_bid = env.book.prev_best_bid
    if best_ask is None:
        best_ask = env.book.prev_best_ask

    if best_bid is None and best_ask is None:
        mark = _mark_price(env)
        best_bid = mark - tick_size
        best_ask = mark + tick_size
    elif best_bid is None:
        best_bid = float(best_ask) - tick_size
    elif best_ask is None:
        best_ask = float(best_bid) + tick_size

    if float(best_ask) <= float(best_bid):
        best_ask = float(best_bid) + tick_size

    return float(best_bid), float(best_ask)


def default_discrete_action_executor(env: "MarketMakingEnv", action: Any) -> None:
    action_id = _normalize_discrete_action(action)
    n_total = DUMP_ACTION_ID + 1

    if action_id < 0 or action_id >= n_total:
        raise ValueError(f"discrete action must be in [0, {n_total - 1}]")

    # Special action: flatten inventory with market order
    if action_id == DUMP_ACTION_ID:
        inv = env.inventory
        if inv > 0:
            fills = env.book.add_market("S", inv, env.book.time)
            for fill in fills:
                env._taker_trades.append(("S", float(fill.price), float(fill.qty)))
        elif inv < 0:
            fills = env.book.add_market("B", abs(inv), env.book.time)
            for fill in fills:
                env._taker_trades.append(("B", float(fill.price), float(fill.qty)))
        return

    # Decode bid/ask styles from the grid (includes "none" style for either side)
    bid_id = action_id // N_QUOTE
    ask_id = action_id % N_QUOTE

    bid_style, bid_qty = QUOTE_ACTIONS[bid_id]
    ask_style, ask_qty = QUOTE_ACTIONS[ask_id]

    # Both "none" → idle, nothing to do
    if bid_style == "none" and ask_style == "none":
        return

    tick_size = 0.01
    best_bid, best_ask = _reference_quotes(env, tick_size=tick_size)

    # Fetch up to 4 anonymous levels so deep_1/2/3 can index L2/L3/L4
    _bid_levels = env.book.top_n_bid_anon(4)  # [(price, qty), ...] best first
    _ask_levels = env.book.top_n_ask_anon(4)

    def _deep_bid_price(depth_idx: int) -> float:
        """Price at anonymous bid level `depth_idx` (1-based from top), fallback to deepest available."""
        if _bid_levels:
            return _bid_levels[min(depth_idx, len(_bid_levels) - 1)][0]
        return best_bid

    def _deep_ask_price(depth_idx: int) -> float:
        """Price at anonymous ask level `depth_idx` (1-based from top), fallback to deepest available."""
        if _ask_levels:
            return _ask_levels[min(depth_idx, len(_ask_levels) - 1)][0]
        return best_ask

    def bid_price(style: str) -> float:
        if style == "best":
            return best_bid + tick_size
        elif style == "level":
            return best_bid
        elif style == "deep_1":
            return _deep_bid_price(1)
        elif style == "deep_2":
            return _deep_bid_price(2)
        elif style == "deep_3":
            return _deep_bid_price(3)
        else:
            raise ValueError(style)

    def ask_price(style: str) -> float:
        if style == "best":
            return best_ask - tick_size
        elif style == "level":
            return best_ask
        elif style == "deep_1":
            return _deep_ask_price(1)
        elif style == "deep_2":
            return _deep_ask_price(2)
        elif style == "deep_3":
            return _deep_ask_price(3)
        else:
            raise ValueError(style)

    _ceil = env.current_max_inventory()
    if bid_style != "none" and env.inventory < _ceil:
        _place_agent_limit_order(env, "B", bid_price(bid_style), float(bid_qty))

    if ask_style != "none" and env.inventory > -_ceil:
        _place_agent_limit_order(env, "S", ask_price(ask_style), float(ask_qty))


# CONTINUOUS ACTION SPACE 
# Discrete part: 5 modes encoding the valid combinations
DROP_ALL_ID = 0
NO_BUY_NO_SELL_ID = 1
BUY_ONLY_ID = 2
SELL_ONLY_ID = 3
BUY_AND_SELL_ID = 4

DISCRETE_MODES = spaces.Discrete(5)

# Continuous part: always 2 dims, executor ignores irrelevant ones
# [buy_price, sell_price]
CONTINUOUS_PARAMS = spaces.Box(
    low=np.array([-np.inf, -np.inf], dtype=np.float32),
    high=np.array([np.inf, np.inf], dtype=np.float32),
)

# Combined
CONT_ACTION_SPACE = spaces.Tuple((DISCRETE_MODES, CONTINUOUS_PARAMS))

def cont_action_executor(env: "MarketMakingEnv", action: Any) -> None:
    mode, continuous = action  # mode: int, continuous: np.array of shape (4,)
    buy_price_raw, sell_price_raw = continuous

    tick_size = 0.01

    best_bid, best_ask = _reference_quotes(env, tick_size=tick_size)

    bid_price = best_bid - buy_price_raw if best_bid - buy_price_raw >= -0.1 else -0.1
    ask_price = sell_price_raw - best_ask if sell_price_raw - best_ask >= -0.1 else -0.1
    

    # see description of continuous action space above
    # Special action: flatten inventory with market order
    if mode == DROP_ALL_ID:
        inv = env.inventory
        if inv > 0:
            fills = env.book.add_market("S", inv, env.book.time)
            for fill in fills:
                env._taker_trades.append(("S", float(fill.price), float(fill.qty)))
        elif inv < 0:
            fills = env.book.add_market("B", abs(inv), env.book.time)
            for fill in fills:
                env._taker_trades.append(("B", float(fill.price), float(fill.qty)))
        return

    # No quote: both sides idle
    if mode == NO_BUY_NO_SELL_ID:
        return

    # Both sides quoting
    _ceil = env.current_max_inventory()
    if mode == BUY_AND_SELL_ID:
        if env.inventory < _ceil:
            _place_agent_limit_order(env, "B", min(bid_price, best_ask), 1)

        if env.inventory > -_ceil:
            _place_agent_limit_order(env, "S", max(ask_price, best_bid), 1)

    if mode == BUY_ONLY_ID:
        if env.inventory < _ceil:
            _place_agent_limit_order(env, "B", min(bid_price, best_ask), 1)

    if mode == SELL_ONLY_ID:
        if env.inventory > -_ceil:
            _place_agent_limit_order(env, "S", max(ask_price, best_bid), 1)


DEFAULT_ACTION_SPACE = spaces.Box(
    low=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    high=np.array([10.0, 10.0, 100.0, 100.0], dtype=np.float32),
    dtype=np.float32,
)

DEFAULT_OBSERVATION_SPACE = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(31,),
    dtype=np.float32,
)

NO_HAWKES_OBSERVATION_SPACE = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(15,),
    dtype=np.float32,
)


class MarketMakingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dt_per_step: float,
        instrument_lifespan_hours: float = 16.5,
        mean_order_lifespan_hours: float = (1/0.061)/3600*3.5, # For the exponential distribution of order lifespans
        auto_cancel: bool = True, # Whether to automatically cancel existing agent orders at the start of each step. If False, the agent must manage its own orders.
        inventory_penalty: float = 0.1,
        terminal_inventory_penalty: float = 25.0,
        obs_fn: Callable[["MarketMakingEnv"], np.ndarray] = default_obs_fn,
        observation_space: Any = DEFAULT_OBSERVATION_SPACE,
        reward_fn: Callable[["MarketMakingEnv", np.ndarray, List[Fill], Dict[str, Any]], float] = default_reward_fn,
        action_space: Any = DEFAULT_ACTION_SPACE,
        action_executor: Callable[["MarketMakingEnv", np.ndarray], None] = default_action_executor,
        initial_cash: float = 0.0,
        max_inventory: int = 100,
        discrete_quote_qty: int = 1,
        include_LO_improve: bool = True,
        use_beta_for_improving_lo_delta: bool = True,
        use_copula_for_non_improving: bool = True,
        use_copula_for_improving: bool = True,
        include_fees_in_reward: int = 1,
        agent_seen_by_non_improving: bool = False,
        agent_seen_by_improving: bool = True,
        reward_fun_type: str = "asymm",
        end_ceiling: Optional[int] = None,
        ceiling_decrease_start_time: Optional[float] = None,
        include_hawkes_in_obs: bool = True,
        dampener: bool = False,
    ):
        if dt_per_step <= 0:
            raise ValueError("dt_per_step must be positive")
        self.include_fees_in_reward = include_fees_in_reward # 1 for yes, 0 for no
        self.agent_seen_by_non_improving = agent_seen_by_non_improving
        self.agent_seen_by_improving = agent_seen_by_improving
        self.use_beta_for_improving_lo_delta = use_beta_for_improving_lo_delta
        self.use_copula_for_non_improving = use_copula_for_non_improving
        self.use_copula_for_improving = use_copula_for_improving
        self.reward_fun_type = reward_fun_type
        self.include_hawkes_in_obs = include_hawkes_in_obs
        self.dampener = dampener
        if reward_fun_type not in ["asymm", "symm", "pnl_only"]:
            raise ValueError("reward_fun_type must be one of 'asymm', 'symm', or 'pnl_only'")

        if self.use_copula_for_non_improving:
            self._copula_bid = _load_copula_arrays('parameters/copula_non_improving_bid.npz')
            self._copula_ask = _load_copula_arrays('parameters/copula_non_improving_bid.npz')
            self._copula_rng = np.random.default_rng()
            # Cap effective window: ~3 sigma is enough for n ~ 6.7M (sigma in rank ~ sqrt(n)/2)
            self._copula_W_eff = int(min(self._copula_bid['W'], self._copula_ask['W'], 3000))
            # Per-env scratch buffer reused across calls (avoid per-call alloc)
            self._copula_scratch = np.empty(2 * self._copula_W_eff, dtype=np.float64)

        if self.use_copula_for_improving:
            self._imp_copula_bid = _load_improving_copula_arrays('parameters/copula_improving_bid.npz')
            self._imp_copula_ask = _load_improving_copula_arrays('parameters/copula_improving_bid.npz')
            self._imp_copula_rng = np.random.default_rng()
            self._imp_copula_W_eff = int(min(self._imp_copula_bid['W'], self._imp_copula_ask['W'], 3000))
            self._imp_copula_scratch = np.empty(2 * self._imp_copula_W_eff, dtype=np.float64)
        
        self.MO_qty_sampler = self._load_MO_qty_sampler()
        self.mo_qty_mixture = self._load_mo_qty_mixture()

        self.dt_per_step = float(dt_per_step)

        ###########################
        
        self.include_LO_improve = include_LO_improve

        if self.include_LO_improve:

            with open(_resolve_resource_path("parameters/hawkes_params_with_improving_SYMMETRIC.pkl"), "rb") as handle:
                self.hawkes_params = pickle.load(handle)
                print("Loaded hawkes_params_with_improving_SYMMETRIC.pkl")
            self.hawkes_history = {"LO_bid_improve": [], "LO_bid_no_improve": [], "LO_ask_improve": [], "LO_ask_no_improve": [], "MO_bid": [], "MO_ask": []} # Empty history initially
        else:
            self.qty_sampler = self._load_sampler("parameters/all_LO_qty.npz")
            self.delta_sampler = self._load_sampler("parameters/all_LO_delta.npz")
            with open(_resolve_resource_path("parameters/hawkes_params.pkl"), "rb") as handle:
                self.hawkes_params = pickle.load(handle)
            self.hawkes_history = {"LO_bid": [], "LO_ask": [], "MO_bid": [], "MO_ask": []} # Empty history initially
        
        self.instrument_lifespan_hours = float(instrument_lifespan_hours)
        self.mean_order_lifespan_hours = float(mean_order_lifespan_hours)

        self.auto_cancel = bool(auto_cancel)
        self.inventory_penalty = float(inventory_penalty)
        self.terminal_inventory_penalty = float(terminal_inventory_penalty)
        self.initial_cash = float(initial_cash)
        self.max_inventory = int(max_inventory)
        self._end_ceiling = int(end_ceiling) if end_ceiling is not None else None
        self._ceiling_decrease_start_time = float(ceiling_decrease_start_time) if ceiling_decrease_start_time is not None else None
        self.discrete_quote_qty = max(1, int(discrete_quote_qty))

        self.obs_fn = obs_fn
        self.reward_fn = reward_fn
        self.action_executor = action_executor

        self.observation_space = observation_space
        self.action_space = action_space

        self.book = Book(
            instrument_lifespan_hours=self.instrument_lifespan_hours,
            mean_order_lifespan_hours=self.mean_order_lifespan_hours,
            dampener=self.dampener,
        )

        self.inventory = 0
        self._prev_inventory = 0
        self.cash = self.initial_cash
        self._prev_pnl = self.cash
        self._last_mid = 0.0
        self.prev_mark_price = _mark_price(self)
        self.fees = []

        self._active_agent_orders: List[int] = []
        self._agent_order_set: set[int] = set()
        self._taker_trades: List[Tuple[str, float, float]] = []
        self._new_agent_order_ids: List[int] = []
        self._improving_lo_qty_rng = np.random.default_rng()
        self._mo_qty_mix_rng = np.random.default_rng()

        # Bookkeeping for running a simulation day
        self.improving_bid_times = []
        self.improving_bid_qtys = []
        self.improving_bid_prices = []
        self.improving_ask_times = []
        self.improving_ask_qtys = []
        self.improving_ask_prices = []
        
        self.non_improving_bid_times = []
        self.non_improving_bid_qtys = []
        self.non_improving_bid_prices = []
        self.non_improving_ask_times = []
        self.non_improving_ask_qtys = []
        self.non_improving_ask_prices = []

        self.buy_mo_times = []
        self.buy_mo_qtys = []
        self.sell_mo_times = []
        self.sell_mo_qtys = []

        self.step_fees = 0.0

        # Pending background events carried over from a block interrupted by an agent fill
        self._pending_events: List[BackgroundEvent] = []
        self._pending_block_end: float = 0.0

        # Used for keeping track of the excitation state of the underlying Hawkes processes
        self.last_decay_time = 0.0
        self.exc_kernel_matrix = np.zeros((6, 6), dtype=np.float64)
        # Construct the alpha and beta matrices
        self.alpha_matrix = self.hawkes_params[1]
        self.beta_matrix = self.hawkes_params[2]


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        del options

        super().reset(seed=seed)

        self.book = Book(
            instrument_lifespan_hours=self.instrument_lifespan_hours,
            mean_order_lifespan_hours=self.mean_order_lifespan_hours,
            seed=seed,
        )
        if self.include_LO_improve:
            self.hawkes_history = {"LO_bid_improve": [], "LO_bid_no_improve": [], "LO_ask_improve": [], "LO_ask_no_improve": [], "MO_bid": [], "MO_ask": []}
        else:
            self.hawkes_history = {"LO_bid": [], "LO_ask": [], "MO_bid": [], "MO_ask": []}

        self.inventory = 0
        self._prev_inventory = 0
        self.cash = self.initial_cash
        self._prev_pnl = self.cash
        self._last_mid = 0.0
        self.prev_mark_price = _mark_price(self)

        self._active_agent_orders.clear()
        self._agent_order_set.clear()
        self._taker_trades.clear()
        self._new_agent_order_ids.clear()
        self._improving_lo_qty_rng = np.random.default_rng(seed)
        self._mo_qty_mix_rng = np.random.default_rng(seed)

        # Clear per-episode bookkeeping lists to prevent cross-episode memory leak
        self.improving_bid_times.clear()
        self.improving_bid_qtys.clear()
        self.improving_bid_prices.clear()
        self.improving_ask_times.clear()
        self.improving_ask_qtys.clear()
        self.improving_ask_prices.clear()
        self.non_improving_bid_times.clear()
        self.non_improving_bid_qtys.clear()
        self.non_improving_bid_prices.clear()
        self.non_improving_ask_times.clear()
        self.non_improving_ask_qtys.clear()
        self.non_improving_ask_prices.clear()
        self.buy_mo_times.clear()
        self.buy_mo_qtys.clear()
        self.sell_mo_times.clear()
        self.sell_mo_qtys.clear()

        self.last_decay_time = 0.0
        self.exc_kernel_matrix[:] = 0.0

        self._pending_events.clear()
        self._pending_block_end = 0.0

        obs = self.obs_fn(self)
        return obs, {}

    def current_max_inventory(self) -> int:
        """Return the effective inventory ceiling at the current book time.

        Linearly interpolates from max_inventory to _end_ceiling between
        _ceiling_decrease_start_time and instrument_lifespan_hours.
        If no end_ceiling was specified, returns max_inventory (backwards-compatible).
        """
        if self._end_ceiling is None:
            return self.max_inventory
        t = self.book.time
        if t <= self._ceiling_decrease_start_time:
            return self.max_inventory
        t_end = self.instrument_lifespan_hours
        if t >= t_end:
            return self._end_ceiling
        frac = (t - self._ceiling_decrease_start_time) / (t_end - self._ceiling_decrease_start_time)
        ceiling = self.max_inventory + frac * (self._end_ceiling - self.max_inventory)
        return max(self._end_ceiling, int(round(ceiling)))

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.book.is_open:
            info = self._build_info([])
            obs = self.obs_fn(self)
            return obs, 0.0, True, False, info
        old_obs = self.obs_fn(self)

        if self.auto_cancel:
            self._cancel_agent_orders()

        self._taker_trades.clear()
        self._new_agent_order_ids.clear()

        self.action_executor(self, action)

        for order_id in self._new_agent_order_ids:
            self._agent_order_set.add(order_id)
            order = self.book.orders.get(order_id)
            if order is not None and order.active and order.qty > 0:
                self._active_agent_orders.append(order_id)

        # Use pending events from a previous interrupted block, or simulate a new block
        if self._pending_events:
            events = self._pending_events
            self._pending_events = []
            block_end = self._pending_block_end
        else:
            t_start = self.book.time
            block_end = min(t_start + self.dt_per_step, self.book.end_time)
            events = self._simulate_background(t_start, block_end)
            self._pending_block_end = block_end

        maker_fills: List[Fill] = []
        early_break = False
        for i, event in enumerate(events):
            event_fills = self._process_background_event(event)
            agent_fill_found = False
            for fill in event_fills:
                if fill.maker_order_id in self._agent_order_set:
                    maker_fills.append(fill)
                    agent_fill_found = True
            if agent_fill_found:
                # Save remaining unprocessed events for the next step
                self._pending_events = events[i + 1:]
                early_break = True
                break

        if not early_break:
            # All events consumed — advance book time to block end
            self._pending_events = []
            if self.book.time < block_end:
                self.book.tick(block_end)

        self._update_from_fills(maker_fills, self._taker_trades)

        combined_fills = list(maker_fills)
        info = self._build_info(combined_fills)

        reward = float(self.reward_fn(self, action, combined_fills, info))
        terminated = not self.book.is_open
        truncated = False

        if terminated and abs(self.inventory) > 0:
            # Force-liquidate remaining inventory at market
            inv = self.inventory
            if inv > 0:
                fills_dump = self.book.add_market("S", float(inv), self.book.time)
                for fill in fills_dump:
                    self._taker_trades.append(("S", float(fill.price), float(fill.qty)))
                    self.inventory -= float(fill.qty)
                    self.cash += float(fill.price) * float(fill.qty)
            elif inv < 0:
                fills_dump = self.book.add_market("B", float(abs(inv)), self.book.time)
                for fill in fills_dump:
                    self._taker_trades.append(("B", float(fill.price), float(fill.qty)))
                    self.inventory += float(fill.qty)
                    self.cash -= float(fill.price) * float(fill.qty)
            reward -= self.terminal_inventory_penalty * float(inv ** 2)

        obs = self.obs_fn(self)
        
        return obs, reward, terminated, truncated, info


    def _simulate_background(self, t_start: float, t_end: float, exc_tau_save: bool = False) -> List[BackgroundEvent]:
        
        if self.include_LO_improve:
            # Trim history lists in-place: discard events older than t_start - 0.02
            # Events are appended in time order, so bisect_left is O(log n).
            cutoff = t_start - 0.02
            hist = self.hawkes_history
            for key in ("LO_bid_improve", "LO_bid_no_improve",
                        "LO_ask_improve", "LO_ask_no_improve",
                        "MO_bid", "MO_ask"):
                lst = hist[key]
                trim = bisect.bisect_left(lst, cutoff)
                if trim > 0:
                    del lst[:trim]

            history = [np.asarray(hist["LO_bid_improve"], dtype=np.float64),
                       np.asarray(hist["LO_bid_no_improve"], dtype=np.float64),
                       np.asarray(hist["LO_ask_improve"], dtype=np.float64),
                       np.asarray(hist["LO_ask_no_improve"], dtype=np.float64),
                       np.asarray(hist["MO_bid"], dtype=np.float64),
                       np.asarray(hist["MO_ask"], dtype=np.float64)]

            param_arrays = self.hawkes_params[0]
            alpha = self.hawkes_params[1]
            beta = self.hawkes_params[2]
            IDA_values = self.hawkes_params[3]
            IDA_config = self.hawkes_params[4]
            baseline = self.hawkes_params[5]

            t_end_clipped = min(self.instrument_lifespan_hours, t_end)
            if exc_tau_save:
                events, exc_taus_save = sim_hawkes(t_start, t_end_clipped, param_arrays, alpha, beta, IDA_values, IDA_config, baseline, history, save_exc_taus=True)
            else:
                events = sim_hawkes(t_start, t_end_clipped, param_arrays, alpha, beta, IDA_values, IDA_config, baseline, history)

            event_type_list = ("LO_bid_improve", "LO_bid_no_improve", "LO_ask_improve", "LO_ask_no_improve", "MO_bid", "MO_ask")

            # Append to history + build BackgroundEvent list in one pass
            backgroundevent_list: List[BackgroundEvent] = []
            for i, etype in enumerate(event_type_list):
                ev_i = events[i]
                hist[etype].extend(ev_i)
                for t in ev_i:
                    backgroundevent_list.append(BackgroundEvent(t, etype))

            backgroundevent_list.sort(key=lambda x: x.time)
            if exc_tau_save:
                return backgroundevent_list, exc_taus_save
            return backgroundevent_list
        else:
            # Trim history lists in-place (same logic as improving branch)
            cutoff = t_start - 0.02
            hist = self.hawkes_history
            for key in ("LO_bid", "LO_ask", "MO_bid", "MO_ask"):
                lst = hist[key]
                trim = bisect.bisect_left(lst, cutoff)
                if trim > 0:
                    del lst[:trim]

            history = [np.asarray(hist[k], dtype=np.float64) for k in
                       ("LO_bid", "LO_ask", "MO_bid", "MO_ask")]

            param_arrays = self.hawkes_params[0]
            alpha = self.hawkes_params[1]
            beta = self.hawkes_params[2]
            IDA_values = self.hawkes_params[3]
            IDA_config = self.hawkes_params[4]
            baseline = self.hawkes_params[5]

            events = sim_hawkes(t_start, min(self.instrument_lifespan_hours, t_end), param_arrays, alpha, beta, IDA_values, IDA_config, baseline, history)

            event_type_list = ("LO_bid", "LO_ask", "MO_bid", "MO_ask")

            backgroundevent_list: List[BackgroundEvent] = []
            for i, etype in enumerate(event_type_list):
                ev_i = events[i]
                hist[etype].extend(ev_i)
                for t in ev_i:
                    backgroundevent_list.append(BackgroundEvent(t, etype))

            backgroundevent_list.sort(key=lambda x: x.time)
            return backgroundevent_list

    # Cancel ALL active orders placed by the agent
    def _cancel_agent_orders(self) -> None:
        for order_id in self._active_agent_orders:
            self.book.cancel_order(order_id)
        self._active_agent_orders.clear()

    # ── LO / MO event handlers ──────────────────────────────────────────

    def _process_improving_lo(self, event: BackgroundEvent, book_side: str) -> List[Fill]:
        """Handle an improving limit order (bid or ask).

        When use_copula_for_improving is True, price and quantity are drawn
        jointly from the improving-LO empirical beta copula conditioned on time.
        The price marginal uses the same Beta parameters as the legacy path;
        the quantity marginal uses the empirical CDF from the copula data.

        When False, falls back to the legacy independent Beta + atom/lognormal
        mixture sampling.
        """
        side = 'bid' if book_side == 'B' else 'ask'
        spread = self.book.spread() if self.agent_seen_by_improving else self.book.spread_ex_agent()

        if self.use_copula_for_improving and spread is not None and spread > 0:
            # ── Copula path ──────────────────────────────────────────────────
            rel_pos, qty = self._sample_improving_lo_copula(event.time)
            improvement = rel_pos * spread   # euros above best_bid / below best_ask

            # if book_side == 'B':
            #     qty *= 1.25
            
            qty = max(0.1, round(qty, 1))

            if event.time < 7:
                improvement_factor = 0.65

            else:
                improvement_factor = 0.35

            if book_side == 'B':
                best = self.book.best_bid() if self.agent_seen_by_improving else self.book.best_bid_ex_agent()
                
                price = best + improvement_factor*improvement
                self.improving_bid_times.append(event.time)
                self.improving_bid_qtys.append(qty)
                self.improving_bid_prices.append(price)
            else:
                best = self.book.best_ask() if self.agent_seen_by_improving else self.book.best_ask_ex_agent()
                price = best - improvement_factor*improvement
                self.improving_ask_times.append(event.time)
                self.improving_ask_qtys.append(qty)
                self.improving_ask_prices.append(price)

            _, fills = self.book.add_limit(book_side, price, qty, event.time)
            self.last_order = f"RelPos: {rel_pos:.4f}, QTY: {qty}, order_type: LO {side} improve copula"
            return fills

        # ── Legacy path ──────────────────────────────────────────────────────
        if spread is not None and self.use_beta_for_improving_lo_delta:
            if self.book.time < 0.5:
                a = 0.304
                b = 37.774
            else:
                a = 0.392
                b = 5.084

            price_delta = -np.round(beta.rvs(a=a, b=b, size=1)[0] * spread * 100, 0)
            price_delta = price_delta
        else:
            price_delta = self._sample_value(
                time_since_start=event.time,
                spread=spread,
                side=side,
                process="LO_improve_delta",
            )
            print("WARNING: NO SPREAD")

        _, qty = self._sample_improving_lo_copula(event.time)

        if book_side == 'B':
            best = self.book.best_bid() if self.agent_seen_by_improving else self.book.best_bid_ex_agent()
            price = best - price_delta / 100
            self.improving_bid_times.append(event.time)
            self.improving_bid_qtys.append(qty)
            self.improving_bid_prices.append(price)
        else:
            best = self.book.best_ask() if self.agent_seen_by_improving else self.book.best_ask_ex_agent()
            price = best + price_delta / 100
            self.improving_ask_times.append(event.time)
            self.improving_ask_qtys.append(qty)
            self.improving_ask_prices.append(price)

        _, fills = self.book.add_limit(book_side, price, qty, event.time)
        self.last_order = f"Delta: {price_delta}, QTY: {qty}, order_type: LO {side} improve"
        return fills

    # ── Improving-LO copula sampler ───────────────────────────────────────────
    def _sample_improving_lo_copula(self, t: float) -> tuple:
        """Return (relative_price_position, qty) from the improving-LO empirical beta copula.

        relative_price_position ∈ (0, 1) is the fractional improvement from the
        current best quote expressed as a share of the spread:
            bid: price = best_bid + rel_pos * spread
            ask: price = best_ask - rel_pos * spread

        The price marginal is inverted via Beta.ppf (parametric; same a/b values
        as the current non-copula code), so the copula captures the joint
        time/price/qty dependence while the Beta shape is preserved.

        Algorithm mirrors _sample_non_improving_copula with the key differences:
          - uses ranks_rpp_sorted instead of ranks_delta_sorted
          - inverts u_rpp through beta.ppf  (not empirical quantile)
          - no sorted_delta lookup
        """
        c   = self._imp_copula_bid #if book_side == 'B' else self._imp_copula_ask
        rng = self._imp_copula_rng

        n                = c['n']
        W                = self._imp_copula_W_eff
        sorted_time_raw  = c['sorted_time_raw']
        log_norm         = c['log_norm']
        r_array          = c['r_array']
        ranks_rpp_sorted = c['ranks_rpp_sorted']
        ranks_qty_sorted = c['ranks_qty_sorted']
        sorted_qty       = c['sorted_qty']

        # 1) Window centre
        j = int(np.searchsorted(sorted_time_raw, np.float32(t), side='right')) - 1
        j = 0 if j < 0 else (n - 1 if j >= n else j)

        lo    = j - W if j > W else 0
        hi    = j + W if j + W < n else n
        width = hi - lo

        # 2) Log-weights (fused linear form; zero per-call allocation via out=)
        u1 = (j + 1) / (n + 1)
        u1 = max(1e-12, min(u1, 1.0 - 1e-12))
        log_u1    = np.log(u1)
        log_1mu1  = np.log(1.0 - u1)
        slope     = log_u1 - log_1mu1
        intercept = n * log_1mu1 - log_u1

        buf = self._imp_copula_scratch[:width]
        np.multiply(r_array[lo:hi], slope, out=buf)
        if intercept != 0.0:
            buf += intercept
        np.subtract(buf, log_norm[lo:hi], out=buf)

        # 3) Softmax → cumsum → inverse-CDF sample
        buf -= buf.max()
        np.exp(buf, out=buf)
        np.cumsum(buf, out=buf)
        i_star  = int(np.searchsorted(buf, rng.random() * buf[-1]))
        i_star  = min(i_star, width - 1)
        abs_idx = lo + i_star

        # 4) Beta copula kernel draws
        r_rpp = int(ranks_rpp_sorted[abs_idx])
        r_qty = int(ranks_qty_sorted[abs_idx])
        u2    = rng.beta(r_rpp, n + 1 - r_rpp)
        u3    = rng.beta(r_qty, n + 1 - r_qty)

        # 5a) Invert relative-price-position via the parametric Beta marginal
        if t < 0.5:
            a_beta, b_beta = 0.304, 37.774
        else:
            a_beta, b_beta = 0.392, 5.084
        rel_pos = float(np.clip(beta.ppf(u2, a_beta, b_beta), 0.0, 1.0))

        # 5b) Invert quantity via empirical quantile
        idx3 = int(u3 * n)
        idx3 = max(0, min(idx3, n - 1))
        qty  = float(sorted_qty[idx3])

        return rel_pos, qty

    # ── Empirical beta copula sampler ─────────────────────────────────────────
    def _sample_non_improving_copula(self, t: float) -> tuple:
        """Return (price_delta_ticks, qty) from the empirical beta copula, conditioning on time t.

        Optimised:
          - Pre-allocated scratch buffer; all numpy ops use out= (zero per-call alloc).
          - r1 is a zero-copy slice of a precomputed r_array.
          - Single fused linear form: log_w = r1 * (log u1 - log(1-u1)) + (n*log(1-u1) - log u1) - log_norm[lo:hi].
          - Categorical sampling via cumsum + searchsorted instead of rng.choice(p=) (~10x faster).
          - Effective window capped to ~3 sigma to keep work O(W_eff) per call.
        """
        c = self._copula_bid #if book_side == 'B' else self._copula_ask
        rng = self._copula_rng

        n        = c['n']
        W        = self._copula_W_eff
        sorted_time_raw    = c['sorted_time_raw']
        log_norm           = c['log_norm']
        r_array            = c['r_array']
        ranks_delta_sorted = c['ranks_delta_sorted']
        ranks_qty_sorted   = c['ranks_qty_sorted']
        sorted_delta       = c['sorted_delta']
        sorted_qty         = c['sorted_qty']

        # 1) Window centre (binary search on raw time).
        # Cast scalar to float32 so np.searchsorted does NOT upcast the entire
        # 6.7M-element float32 array to float64 (~10 ms hidden cost otherwise).
        j = int(np.searchsorted(sorted_time_raw, np.float32(t), side='right')) - 1
        if j < 0:
            j = 0
        elif j >= n:
            j = n - 1

        lo = j - W if j > W else 0
        hi = j + W if j + W < n else n
        width = hi - lo

        # 2) Linearised log-weights into pre-allocated buffer
        u1 = (j + 1) / (n + 1)
        if u1 < 1e-12:
            u1 = 1e-12
        elif u1 > 1.0 - 1e-12:
            u1 = 1.0 - 1e-12
        log_u1   = np.log(u1)
        log_1mu1 = np.log(1.0 - u1)
        slope    = log_u1 - log_1mu1
        intercept = n * log_1mu1 - log_u1

        buf = self._copula_scratch[:width]
        np.multiply(r_array[lo:hi], slope, out=buf)
        if intercept != 0.0:
            buf += intercept
        np.subtract(buf, log_norm[lo:hi], out=buf)

        # 3) Softmax (in place) -> cumulative sum -> inverse-CDF sample
        buf -= buf.max()
        np.exp(buf, out=buf)
        np.cumsum(buf, out=buf)
        target = rng.random() * buf[-1]
        i_star = int(np.searchsorted(buf, target))
        if i_star >= width:
            i_star = width - 1
        abs_idx = lo + i_star

        # 4) Beta draws for delta + qty conditional on chosen mixture component
        rd = int(ranks_delta_sorted[abs_idx])
        rq = int(ranks_qty_sorted[abs_idx])
        u2 = rng.beta(rd, n + 1 - rd)
        u3 = rng.beta(rq, n + 1 - rq)

        # 5) Empirical-quantile inversion
        idx2 = int(u2 * n)
        if idx2 >= n:
            idx2 = n - 1
        elif idx2 < 0:
            idx2 = 0
        idx3 = int(u3 * n)
        if idx3 >= n:
            idx3 = n - 1
        elif idx3 < 0:
            idx3 = 0

        return float(sorted_delta[idx2]), 1.4 * float(sorted_qty[idx3])

    def _process_non_improving_lo_copula(self, event: BackgroundEvent, book_side: str) -> List[Fill]:
        """Handle a non-improving limit order using the empirical beta copula sampler."""
        side = 'bid' if book_side == 'B' else 'ask'

        price_delta, qty = self._sample_non_improving_copula(event.time)
        if price_delta < 0:
            print(f"WARNING: negative price_delta {price_delta} from copula sampler; setting to 0")
            raise ValueError("negative price_delta from copula sampler")
        # if side == 'bid':
        #     qty *= 1.3
        # else:
        #     qty *= 1.1
        
        # If we are in the first 20 minutes, cap the price delta to avoid the instant jump in price
        if event.time < 20/60:
            price_delta = min(price_delta, 1000.0)
        
        # --- Price level ---
        if book_side == 'B':
            best = self.book.best_bid() if self.agent_seen_by_non_improving else self.book.best_bid_ex_agent()
            price = best - price_delta / 100
            self.non_improving_bid_times.append(event.time)
            self.non_improving_bid_qtys.append(qty)
            self.non_improving_bid_prices.append(price)
        else:
            best = self.book.best_ask() if self.agent_seen_by_non_improving else self.book.best_ask_ex_agent()
            price = best + price_delta / 100
            self.non_improving_ask_times.append(event.time)
            self.non_improving_ask_qtys.append(qty)
            self.non_improving_ask_prices.append(price)

        qty = max(0.1, round(float(qty), 1))
        _, fills = self.book.add_limit(book_side, price, qty, event.time)
        self.last_order = f"Delta: {price_delta}, QTY: {qty}, order_type: LO {side} no_improve copula"
        return fills

    def _process_mo(self, event: BackgroundEvent, book_side: str) -> List[Fill]:
        """Handle a market order (bid or ask)."""
        side = 'bid'# if book_side == 'B' else 'ask'
        qty = self._sample_mo_qty_mixture(side=side)
        # if side == 'bid':
        #     qty *= 1.25
        # else:
        #     qty *= 0.95
        
        # Round qty to nearest 0.1 and ensure it's at least 0.1
        #qty = max(0.1, round(float(qty), 1))

        self.last_order = f"QTY: {qty}, order_type: MO {side}"

        if book_side == 'B':
            self.buy_mo_times.append(event.time)
            self.buy_mo_qtys.append(qty)

        else:
            self.sell_mo_times.append(event.time)
            self.sell_mo_qtys.append(qty)


        return self.book.add_market(book_side, qty, event.time)

    # ── Main dispatcher ──────────────────────────────────────────────────

    # Map event type → column index for Hawkes excitation kernel update
    _EVENT_COL = {
        "LO_bid_improve": 0,
        "LO_bid_no_improve": 1,
        "LO_ask_improve": 2,
        "LO_ask_no_improve": 3,
        "MO_bid": 4,
        "MO_ask": 5,
    }

    def _process_background_event(self, event: BackgroundEvent) -> List[Fill]:
        if event.time < self.book.time:
            raise ValueError("background event time must be non-decreasing")

        # Advance clock BEFORE querying best prices so expired orders are removed first
        self.book.tick(event.time)

        etype = event.event_type

        if self.include_LO_improve:
            if etype == "LO_bid_improve":
                to_return = self._process_improving_lo(event, "B")
            elif etype == "LO_ask_improve":
                to_return = self._process_improving_lo(event, "S")
            elif etype == "LO_bid_no_improve":
                if self.use_copula_for_non_improving:
                    to_return = self._process_non_improving_lo_copula(event, "B")
                else:
                    to_return = self._process_non_improving_lo(event, "B")
            elif etype == "LO_ask_no_improve":
                if self.use_copula_for_non_improving:
                    to_return = self._process_non_improving_lo_copula(event, "S")
                else:
                    to_return = self._process_non_improving_lo(event, "S")
            elif etype == "MO_bid":
                to_return = self._process_mo(event, "B")
            elif etype == "MO_ask":
                to_return = self._process_mo(event, "S")
            else:
                to_return = []
        else:
            if etype == "LO_bid":
                to_return = self._process_simple_lo(event, "B")
            elif etype == "LO_ask":
                to_return = self._process_simple_lo(event, "S")
            elif etype == "MO_bid":
                to_return = self._process_mo(event, "B")
            elif etype == "MO_ask":
                to_return = self._process_mo(event, "S")
            else:
                to_return = []

        # Update excitation kernel: add alpha column for this event type, then decay
        col = self._EVENT_COL.get(etype)
        if col is not None:
            self.exc_kernel_matrix[:, col] += self.alpha_matrix[:, col]
        self.exc_kernel_matrix *= np.exp(-self.beta_matrix * (event.time - self.last_decay_time))
        self.last_decay_time = event.time

        return to_return


    def _update_from_fills(
        self,
        maker_fills: Sequence[Fill],
        taker_trades: Sequence[Tuple[str, float, float]],
    ) -> None:
        for fill in maker_fills:
            maker_order = self.book.orders.get(fill.maker_order_id)
            if maker_order is None:
                continue
            side = maker_order.side
            qty = float(fill.qty)
            price = float(fill.price)

            if side == "B":
                self.inventory += qty
                self.cash -= price * qty
            else:
                self.inventory -= qty
                self.cash += price * qty

        for side, price, qty in taker_trades:
            q = float(qty)
            p = float(price)
            if side == "B":
                self.inventory += q
                self.cash -= p * q
            else:
                self.inventory -= q
                self.cash += p * q

    """
    Method to build state vector
    May change the things returned here
    """
    def _build_info(self, fills: List[Fill]) -> Dict[str, Any]:
        mark = _mark_price(self)
        pnl = self.cash + self.inventory * mark
        return {
            "fills": fills,
            "inventory": self.inventory,
            "cash": self.cash,
            "mark_price": mark,
            "step_fees": calc_fees(self, fills),
            "pnl": pnl,
            "book_time": self.book.time,
            "time_remaining": max(0.0, self.book.end_time - self.book.time),
        }
    

    """
    Function to sample a quantity:
    Takes time_since_start, spread and side (Bid/Ask) as input and returns a 
    quantity sampled from the corresponding distribution in the npz file.
    """
    def _sample_quantity(self, time_since_start, spread, side):
        """Sample one quantity conditional on time, spread, and side."""
        t = np.searchsorted(self.qty_sampler['time_edges'], float(time_since_start), side='right') - 1
        t = int(np.clip(t, 0, self.qty_sampler['n_time_bins'] - 1))

        s_edges = self.qty_sampler['spread_edges'][t]
        s = np.searchsorted(s_edges, float(spread), side='right') - 1
        s = int(np.clip(s, 0, self.qty_sampler['n_spread_bins'] - 1))

        cell = t * self.qty_sampler['n_spread_bins'] + s

        side_l = str(side).lower()
        if side_l in ('bid', 'buy', 'b'):
            values = self.qty_sampler['bid_values']
            offsets = self.qty_sampler['bid_offsets']
            fallback = self.qty_sampler['bid_pool']
        elif side_l in ('ask', 'sell', 'a'):
            values = self.qty_sampler['ask_values']
            offsets = self.qty_sampler['ask_offsets']
            fallback = self.qty_sampler['ask_pool']
        else:
            raise ValueError("side must be one of: 'bid', 'ask', 'buy', 'sell', 'b', 'a'.")

        start = int(offsets[cell])
        end = int(offsets[cell + 1])

        if end > start:
            idx = self.qty_sampler['rng'].integers(start, end)
            return float(values[idx])

        if fallback.size == 0:
            raise ValueError('No quantity data available to sample from for this side.')

        idx = self.qty_sampler['rng'].integers(0, fallback.size)
        return float(fallback[idx])

    ########## 

    def _sample_value(self, process, time_since_start, spread, side):
        sampler = self.samplers[process]
        """Sample one value given current time, spread, and side ('bid'/'ask' or 'buy'/'sell')."""
        t_edges = sampler['time_edges']
        t = np.searchsorted(t_edges, float(time_since_start), side='right') - 1
        t = int(np.clip(t, 0, sampler['n_time_bins'] - 1))

        s_edges = sampler['spread_edges'][t]
        s = np.searchsorted(s_edges, float(spread), side='right') - 1
        s = int(np.clip(s, 0, sampler['n_spread_bins'] - 1))

        cell = t * sampler['n_spread_bins'] + s

        side_l = str(side).lower()
        if side_l in ('bid', 'buy', 'b'):
            values = sampler['bid_values']
            offsets = sampler['bid_offsets']
            fallback = sampler['bid_pool']
        elif side_l in ('ask', 'sell', 'a'):
            values = sampler['ask_values']
            offsets = sampler['ask_offsets']
            fallback = sampler['ask_pool']
        else:
            raise ValueError("side must be one of: 'bid', 'ask', 'buy', 'sell', 'b', 'a'.")

        start = int(offsets[cell])
        end = int(offsets[cell + 1])

        if end > start:
            idx = sampler['rng'].integers(start, end)
            return float(values[idx])

        if fallback.size == 0:
            raise ValueError('No data available to sample from for this side.')

        idx = sampler['rng'].integers(0, fallback.size)
        return float(fallback[idx])


    """
    Function to load the npz file containing the MO sampler distributions
    """
    def _load_MO_qty_sampler(self, path='parameters/quantity_sampler_MO.npz', seed=None):
        """Load sampler data once and keep returned object in memory for simulation."""
        d = np.load(_resolve_resource_path(path))
        return {
            'n_time_bins': int(d['n_time_bins']),
            'n_spread_bins': int(d['n_spread_bins']),
            'time_edges': d['time_edges'],
            'spread_edges': d['spread_edges'],
            'bid_qty_values': d['bid_qty_values'],
            'bid_qty_offsets': d['bid_qty_offsets'],
            'ask_qty_values': d['ask_qty_values'],
            'ask_qty_offsets': d['ask_qty_offsets'],
            'bid_qty_pool': d['bid_qty_pool'],
            'ask_qty_pool': d['ask_qty_pool'],
            'rng': np.random.default_rng(seed),
        }


    """
    Function to sample a MO quantity for an incoming order:
    Takes time_since_start, spread and side (Bid/Ask) as input and returns a MO qty sampled from the 
    corresponding distribution in the npz file.
    """
    def _sample_MO_qty(self, time_since_start, spread, side):
        """Sample one MO qty given current time, spread, and side ('bid'/'ask' or 'buy'/'sell')."""
        t_edges = self.MO_qty_sampler['time_edges']
        t = np.searchsorted(t_edges, float(time_since_start), side='right') - 1
        t = int(np.clip(t, 0, self.MO_qty_sampler['n_time_bins'] - 1))

        s_edges = self.MO_qty_sampler['spread_edges'][t]
        s = np.searchsorted(s_edges, float(spread), side='right') - 1
        s = int(np.clip(s, 0, self.MO_qty_sampler['n_spread_bins'] - 1))

        cell = t * self.MO_qty_sampler['n_spread_bins'] + s

        side_l = str(side).lower()
        if side_l in ('bid', 'buy', 'b'):
            values = self.MO_qty_sampler['bid_qty_values']
            offsets = self.MO_qty_sampler['bid_qty_offsets']
            fallback = self.MO_qty_sampler['bid_qty_pool']
        elif side_l in ('ask', 'sell', 'a'):
            values = self.MO_qty_sampler['ask_qty_values']
            offsets = self.MO_qty_sampler['ask_qty_offsets']
            fallback = self.MO_qty_sampler['ask_qty_pool']
        else:
            raise ValueError("side must be one of: 'bid', 'ask', 'buy', 'sell', 'b', 'a'.")

        start = int(offsets[cell])
        end = int(offsets[cell + 1])

        if end > start:
            idx = self.MO_qty_sampler['rng'].integers(start, end)
            return float(values[idx])

        if fallback.size == 0:
            raise ValueError('No data available to sample from for this side.')

        idx = self.MO_qty_sampler['rng'].integers(0, fallback.size)
        return float(fallback[idx])

    def _load_mo_qty_mixture(self, path='parameters/mo_qty_mixture.npz'):
        """Load the atom+lognormal mixture parameters for MO quantity sampling."""
        d = np.load(_resolve_resource_path(path))
        return {
            'atoms':          d['atoms'],
            'buy_atom_probs': d['buy_atom_probs'],
            'buy_p_rem':      float(d['buy_p_rem']),
            'buy_shape':      float(d['buy_shape']),
            'buy_scale':      float(d['buy_scale']),
            'sell_atom_probs': d['sell_atom_probs'],
            'sell_p_rem':     float(d['sell_p_rem']),
            'sell_shape':     float(d['sell_shape']),
            'sell_scale':     float(d['sell_scale']),
        }

    def _sample_mo_qty_mixture(self, side: str) -> float:
        """Sample one MO quantity from the atom+lognormal mixture."""
        d = self.mo_qty_mixture
        

        atom_probs = d['buy_atom_probs']
        shape      = d['buy_shape']
        scale      = d['buy_scale']

        # side_l = str(side).lower()
        # if side_l in ('bid', 'buy', 'b'):
        #     atom_probs = d['buy_atom_probs']
        #     shape      = d['buy_shape']
        #     scale      = d['buy_scale']
        # elif side_l in ('ask', 'sell', 'a', 's'):
        #     atom_probs = d['sell_atom_probs']
        #     shape      = d['sell_shape']
        #     scale      = d['sell_scale']
        # else:
        #     raise ValueError(f"Unknown side: {side}")

        atoms = d['atoms']
        u = self._mo_qty_mix_rng.random()
        cum_atoms = np.cumsum(atom_probs)
        idx = np.searchsorted(cum_atoms, u, side='right')

        if idx < len(atoms):
            return float(atoms[idx])

        # Continuous remainder: lognormal
        qty = self._mo_qty_mix_rng.lognormal(
            mean=float(np.log(scale)), sigma=float(shape)
        )
        return max(0.1, float(qty))

    def _add_agent_order_to_hawkes_hist(self, time, order_type):
        self.hawkes_history[order_type].append(time)

    def _book_side_snapshot(self, side: str, depth: int = 20) -> List[Dict[str, float]]:
        """Capture top levels for one side with volume.

        The new Book stores tallied volume per tick (no per-order iteration),
        so per-order time-to-expiry is not available for anonymous volume.
        """
        book = self.book
        levels: List[Dict[str, float]] = []

        if side == "B":
            it = reversed(book.bids.items())
        else:
            it = iter(book.asks.items())

        for tick, volume in it:
            if volume <= 0:
                continue
            levels.append(
                {
                    "price": book._to_price(tick),
                    "volume": float(volume),
                }
            )
            if len(levels) >= depth:
                break

        return levels

    def _capture_book_snapshot(self, depth: int = 20) -> Dict[str, Any]:
        return {
            "time": float(self.book.time),
            "bids": self._book_side_snapshot("B", depth=depth),
            "asks": self._book_side_snapshot("S", depth=depth),
        }
    
    def simulate_market_no_impact_version(self):
        best_bid_save = []
        best_ask_save = []
        states = []
        time_save = []

        time_sim = 0
        dt = self.dt_per_step

        states.append(default_obs_fn(self))

        while time_sim <= self.instrument_lifespan_hours:
            order_arrival, exc_taus = self._simulate_background(time_sim, time_sim + dt, exc_tau_save = True)
            for i in range(len(order_arrival)):
                self._process_background_event(order_arrival[i])
                best_bid_save.extend([self.book.prev_best_bid])
                best_ask_save.extend([self.book.prev_best_ask])
                time_save.extend([order_arrival[i].time])

            time_sim += dt
            state = default_obs_fn(self)
            states.append(state)

        prices_and_qtys={
            
            'buy_mo': {
                'times': self.buy_mo_times,
                'qtys': self.buy_mo_qtys
            },
            'sell_mo': {
                'times': self.sell_mo_times,
                'qtys': self.sell_mo_qtys
            },
        }

        return best_bid_save, best_ask_save, time_save, prices_and_qtys, states 
    
    def simulate_market(self, save_exc_taus: bool = False):
        best_bid_save = []
        best_ask_save = []
        states = []
        time_save = []

        book_snapshots = []
        exc_taus_to_save = [[] for _ in range(6)]

        time_sim = 0
        dt = self.dt_per_step

        extra_time_steps = 0

        states.append(default_obs_fn(self))

        while time_sim <= self.instrument_lifespan_hours:
            #print(time_sim)
            if save_exc_taus:
                order_arrival, exc_taus = self._simulate_background(time_sim, time_sim + dt, exc_tau_save = True)
                for old, new in zip(exc_taus_to_save, exc_taus):
                    old.extend(new)
            else:
                order_arrival = self._simulate_background(time_sim, time_sim + dt)
            
            

            # Get the prices and volume at the best five levels of the OB
            old_best_ask = self.book.prev_best_ask
            old_best_bid = self.book.prev_best_bid
            print_counter = 0
            for i in range(len(order_arrival)):
                # --- Capture snapshots at any order expirations before this event ---
                expiry_times = self.book.pending_expiry_times(order_arrival[i].time)
                for t_exp in expiry_times:
                    extra_time_steps += 1
                    self.book.tick(t_exp)
                    if abs(self.book.prev_best_ask - old_best_ask) > 10:
                        print("BIG JUMP IN BEST ASK (expiry)")
                        print(t_exp)
                        print("jump: ", self.book.prev_best_ask - old_best_ask)
                    if abs(self.book.prev_best_bid - old_best_bid) > 10:
                        print("BIG JUMP IN BEST BID (expiry)")
                        print(t_exp)
                        print("jump: ", self.book.prev_best_bid - old_best_bid)
                    best_bid_save.append(self.book.prev_best_bid)
                    best_ask_save.append(self.book.prev_best_ask)
                    time_save.append(t_exp)
                    old_best_ask = self.book.prev_best_ask
                    old_best_bid = self.book.prev_best_bid

                self._process_background_event(order_arrival[i])
                
                
                
                # print_counter += 1
                # if print_counter >= 1000:
                #     #print(self.book.time)
                #     print_counter = 0
                if abs(self.book.prev_best_ask - old_best_ask) > 10:
                    print("BIG JUMP IN BEST ASK")
                    print(order_arrival[i].time)
                    print("jump: ", self.book.prev_best_ask - old_best_ask)
                    print(order_arrival[i].event_type)
                    print(self.last_order)
                
                if abs(self.book.prev_best_bid - old_best_bid) > 10:
                    print("BIG JUMP IN BEST BID")
                    print(order_arrival[i].time)
                    print("jump: ", self.book.prev_best_bid - old_best_bid)
                    print(order_arrival[i].event_type)
                    print(self.last_order)
                best_bid_save.extend([self.book.prev_best_bid])
                best_ask_save.extend([self.book.prev_best_ask])
                time_save.extend([order_arrival[i].time])
                #book_snapshots.append(self._capture_book_snapshot(depth=5))
                old_best_ask = self.book.prev_best_ask
                old_best_bid = self.book.prev_best_bid

            time_sim += dt
            # Collect state (with inventory set to 0 since we don't observe it)
            state = default_obs_fn(self)
            states.append(state)

        prices_and_qtys={
            
            'improving_bid': {
                'times': self.improving_bid_times,
                'qtys': self.improving_bid_qtys,
                'prices': self.improving_bid_prices
            },
            'improving_ask': {
                'times': self.improving_ask_times,
                'qtys': self.improving_ask_qtys,
                'prices': self.improving_ask_prices
            },
            'non_improving_bid': {
                'times': self.non_improving_bid_times,
                'qtys': self.non_improving_bid_qtys,
                'prices': self.non_improving_bid_prices
            },
            'non_improving_ask': {
                'times': self.non_improving_ask_times,
                'qtys': self.non_improving_ask_qtys,
                'prices': self.non_improving_ask_prices
            },
            'buy_mo': {
                'times': self.buy_mo_times,
                'qtys': self.buy_mo_qtys
            },
            'sell_mo': {
                'times': self.sell_mo_times,
                'qtys': self.sell_mo_qtys
            },
            'book_snapshots': book_snapshots,
        }

        print(f"Total time steps: {len(best_bid_save)}")
        print(f"Total extra time steps from expiries: {extra_time_steps}")
        if save_exc_taus:
            return best_bid_save, best_ask_save, time_save, prices_and_qtys, exc_taus_to_save, states

        
        return best_bid_save, best_ask_save, time_save, prices_and_qtys

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    environment_test = MarketMakingEnv(0.49)
    # environment_test._simulate_background(0,16.5)
    # test_sim = environment_test.hawkes_history["LO_bid"]
    # plt.hist(test_sim, bins = 50)
    # plt.show()
    best_bid_list, best_ask_list, time_save, prices_and_qtys = environment_test.simulate_market()
    
    print(len(best_bid_list))

    bid_times_simple = np.linspace(0, 16.5, len(best_bid_list))
    ask_times_simple = np.linspace(0, 16.5, len(best_ask_list))
    plt.plot(bid_times_simple, best_bid_list, color = "red")
    plt.plot(ask_times_simple, best_ask_list, color = "green")
    plt.show()

