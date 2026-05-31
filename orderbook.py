
"""
Fast price-priority limit order book.

Drops time priority entirely for anonymous flow:
  - Anonymous resting volume is stored as a single scalar per integer-tick
    price level (no per-order bookkeeping for anonymous orders).
  - Each anonymous LO arrival still carries a sampled lifetime; expiry is
    handled by a global min-heap of (t_expiry, tick, qty, side) cohorts that
    decrement the level's anonymous volume when popped.

Agent orders are tracked individually so the RL environment can identify
its own fills and cancel by id. At any matched level, agent orders are
filled FIRST (in order of insertion), then anonymous volume.

No iceberg orders. No legacy random-cancel. Prices are stored internally
as int ticks (1 tick = ``tick_size`` €, default 0.01).
"""

from __future__ import annotations

import heapq
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import t as student_t
from sortedcontainers import SortedDict

try:
    from .lifetime_sampler import (
        load_exponential_bin_sampler,
        load_exponential_bin_sampler_2d,
    )
except ImportError:  # script / notebook execution
    from lifetime_sampler import (  # type: ignore
        load_exponential_bin_sampler,
        load_exponential_bin_sampler_2d,
    )


# Sentinel maker_order_id used for fills against anonymous (untracked) volume.
# `_next_id` starts at 1, so this never collides with a real agent order id.
ANON_MAKER_ID = 0


@dataclass
class Fill:
    taker_side: str          # "B" or "S"
    maker_order_id: int      # 0 == anonymous volume; >0 == agent order id
    price: float
    qty: float


@dataclass
class RestingOrder:
    order_id: int
    side: str                # "B" or "S"
    price: float             # human-readable price (float €), for compatibility
    tick: int                # internal int-tick representation
    qty: float
    expiry_time: float
    active: bool = True


class Book:
    """
    Continuous-time limit order book WITHOUT price-time priority.

    See module docstring for the design contract.
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        instrument_lifespan_hours: float = 16.5,
        mean_order_lifespan_hours: float = (1 / 0.061) / 3600 * 20,
        prev_best_bid: Optional[float] = None,
        prev_best_ask: Optional[float] = None,
        seed: Optional[int] = None,
        lifetime_sampler: Optional[object] = None,
        lifetime_sampler_config: Optional[str] = "parameters/lifetime_exp_2d_10x10.json",
        tick_size: float = 0.01,
        dampener: bool = False,
    ) -> None:
        if instrument_lifespan_hours <= 0:
            raise ValueError("instrument_lifespan_hours must be positive")
        if mean_order_lifespan_hours <= 0:
            raise ValueError("mean_order_lifespan_hours must be positive")
        if tick_size <= 0:
            raise ValueError("tick_size must be positive")

        self._tick_size = float(tick_size)
        self._dampener = bool(dampener)


        # Total volume per tick (anon + agent), one SortedDict per side.
        self.bids: SortedDict = SortedDict()  # int tick -> float total qty
        self.asks: SortedDict = SortedDict()

        # Anonymous-only volume per tick (used for the agent-first split in _match).
        self._anon_bid: Dict[int, float] = {}
        self._anon_ask: Dict[int, float] = {}

        # Agent orders per side per tick. OrderedDict preserves insertion order
        # so agent-first matching at a level is deterministic FIFO.
        self._agent_bid_at_level: Dict[int, "OrderedDict[int, RestingOrder]"] = {}
        self._agent_ask_at_level: Dict[int, "OrderedDict[int, RestingOrder]"] = {}

        # Global agent order registry for O(1) cancel / lookup.
        self.agent_orders: Dict[int, RestingOrder] = {}

        # Running side totals for O(1) total-volume access in _compute_priority_index.
        self._side_total_bid: float = 0.0
        self._side_total_ask: float = 0.0

        # Expiry heaps.
        # Anonymous: (t_expiry, tick, qty, side)
        self._anon_expiry: List[Tuple[float, int, float, str]] = []
        # Agent: (t_expiry, order_id)
        self._agent_expiry: List[Tuple[float, int]] = []

        self._next_id: int = 1

        self.time: float = 0.0
        self.end_time: float = float(instrument_lifespan_hours)
        self.mean_order_lifespan_hours: float = float(mean_order_lifespan_hours)

        self.rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)


        # Initial best bid / ask. Match the student-t draw used by
        # market_simulation.OrderBook so the two books are interchangeable for
        # initial conditions when callers don't override.
        if prev_best_bid is None or prev_best_ask is None:
            ida_1_price_sample = student_t.rvs(
                df=2.6216111915867244,
                loc=99.26534615310835,
                scale=19.857199896024298,
                size=1,
            )[0]
            spread_sample = 40
            self.prev_best_bid = (
                float(prev_best_bid)
                if prev_best_bid is not None
                else float(ida_1_price_sample - 0.5 * spread_sample)
            )
            self.prev_best_ask = (
                float(prev_best_ask)
                if prev_best_ask is not None
                else float(ida_1_price_sample + 0.5 * spread_sample)
            )
        else:
            self.prev_best_bid = float(prev_best_bid)
            self.prev_best_ask = float(prev_best_ask)

        # Agent-excluded prev trackers (never contaminated by agent quotes)
        self.prev_best_bid_ex_agent = self.prev_best_bid
        self.prev_best_ask_ex_agent = self.prev_best_ask

        # Lifetime sampler: same selection logic as market_simulation.OrderBook.
        if lifetime_sampler is not None:
            self._lifetime_sampler = lifetime_sampler
        elif lifetime_sampler_config is not None:
            config_path = Path(lifetime_sampler_config)
            if not config_path.is_absolute():
                config_path = Path(__file__).resolve().parent / config_path
            config_path = config_path.resolve()
            with open(config_path, "r") as f:
                _peek = json.load(f)
            if "rates_2d" in _peek:
                self._lifetime_sampler = load_exponential_bin_sampler_2d(config_path)
            else:
                self._lifetime_sampler = load_exponential_bin_sampler(config_path)
        else:
            self._lifetime_sampler = None

    # ----------------------------------------------------------- properties

    @property
    def is_open(self) -> bool:
        return self.time < self.end_time

    @property
    def orders(self) -> Dict[int, RestingOrder]:
        """Compatibility alias for code that reads ``book.orders[id]`` for agent orders."""
        return self.agent_orders

    # --------------------------------------------------------- tick helpers

    def _to_tick(self, price: float) -> int:
        # Round-to-nearest tick; matches what the env expects from offset arithmetic.
        return int(round(float(price) / self._tick_size))

    def _to_price(self, tick: int) -> float:
        return tick * self._tick_size

    # ----------------------------------------------------------- time mgmt

    def _sync_to_event_time(self, t: float) -> None:
        if t < self.time:
            raise ValueError(
                f"event time {t} is earlier than current book time {self.time}"
            )
        if t > self.end_time:
            raise RuntimeError(
                f"event time {t} is beyond instrument end time {self.end_time}"
            )
        self.time = t
        self._expire()

    def _expire(self) -> None:
        """Drain both expiry heaps up to ``self.time``."""
        now = self.time

        # --- Anonymous cohorts ---
        anon_heap = self._anon_expiry
        while anon_heap and anon_heap[0][0] <= now:
            _, tick, qty, side = heapq.heappop(anon_heap)
            if qty <= 0.0:
                continue
            if side == "B":
                cur = self._anon_bid.get(tick, 0.0)
                if cur <= 0.0:
                    continue
                take = qty if qty <= cur else cur
                new = cur - take
                if new <= 1e-12:
                    self._anon_bid.pop(tick, None)
                else:
                    self._anon_bid[tick] = new
                self._dec_level("B", tick, take)
            else:
                cur = self._anon_ask.get(tick, 0.0)
                if cur <= 0.0:
                    continue
                take = qty if qty <= cur else cur
                new = cur - take
                if new <= 1e-12:
                    self._anon_ask.pop(tick, None)
                else:
                    self._anon_ask[tick] = new
                self._dec_level("S", tick, take)

        # --- Agent orders ---
        ag_heap = self._agent_expiry
        orders = self.agent_orders
        while ag_heap and ag_heap[0][0] <= now:
            t_exp, oid = heapq.heappop(ag_heap)
            order = orders.get(oid)
            if order is None or not order.active or order.qty <= 0:
                continue
            if order.expiry_time != t_exp:
                continue
            self._remove_agent_order(order)

    def _close_if_needed(self) -> None:
        if self.time < self.end_time:
            return
        for o in self.agent_orders.values():
            o.active = False
            o.qty = 0
        self.bids.clear()
        self.asks.clear()
        self._anon_bid.clear()
        self._anon_ask.clear()
        self._agent_bid_at_level.clear()
        self._agent_ask_at_level.clear()
        self._anon_expiry.clear()
        self._agent_expiry.clear()
        self._side_total_bid = 0.0
        self._side_total_ask = 0.0

    def tick(self, t: float) -> None:
        """Advance the clock without placing an order."""
        self._sync_to_event_time(t)
        self._close_if_needed()

    def pending_expiry_times(self, before: float) -> List[float]:
        """Return sorted unique expiry times in (self.time, before].

        Read-only peek at both expiry heaps — nothing is popped or expired.
        Used by ``simulate_market`` to step the book to each expiry instant
        and record a snapshot of the best bid/ask at that moment.
        """
        times: set = set()
        now = self.time
        for entry in self._anon_expiry:
            t_exp = entry[0]
            if t_exp > now and t_exp <= before:
                times.add(t_exp)
        for entry in self._agent_expiry:
            t_exp = entry[0]
            if t_exp > now and t_exp <= before:
                times.add(t_exp)
        return sorted(times)

    # ------------------------------------------------------- level helpers

    def _inc_level(self, side: str, tick: int, qty: float) -> None:
        if qty <= 0:
            return
        book = self.bids if side == "B" else self.asks
        book[tick] = book.get(tick, 0.0) + qty
        if side == "B":
            self._side_total_bid += qty
        else:
            self._side_total_ask += qty

    def _dec_level(self, side: str, tick: int, qty: float) -> None:
        if qty <= 0:
            return
        book = self.bids if side == "B" else self.asks
        cur = book.get(tick, 0.0)
        new = cur - qty
        if new <= 1e-12:
            book.pop(tick, None)
        else:
            book[tick] = new
        if side == "B":
            self._side_total_bid -= qty
            if self._side_total_bid < 0.0:
                self._side_total_bid = 0.0
        else:
            self._side_total_ask -= qty
            if self._side_total_ask < 0.0:
                self._side_total_ask = 0.0

    def _remove_agent_order(self, order: RestingOrder) -> None:
        """Fully remove an agent order from registries and decrement level total."""
        side = order.side
        tick = order.tick
        qty = order.qty
        order.active = False
        order.qty = 0
        registry = (
            self._agent_bid_at_level if side == "B" else self._agent_ask_at_level
        )
        level = registry.get(tick)
        if level is not None:
            level.pop(order.order_id, None)
            if not level:
                registry.pop(tick, None)
        if qty > 0:
            self._dec_level(side, tick, qty)

    # ------------------------------------------------------- best/quote API

    def best_bid(self) -> Optional[float]:
        self._expire()
        if self.bids:
            tick = self.bids.peekitem(-1)[0]
            price = self._to_price(tick)
            self.prev_best_bid = price
            return price
        return self.prev_best_bid

    def best_ask(self) -> Optional[float]:
        self._expire()
        if self.asks:
            tick = self.asks.peekitem(0)[0]
            price = self._to_price(tick)
            self.prev_best_ask = price
            return price
        return self.prev_best_ask

    def best_bid_qty(self) -> float:
        self._expire()
        if self.bids:
            return float(self.bids.peekitem(-1)[1])
        return 0.0

    def best_ask_qty(self) -> float:
        self._expire()
        if self.asks:
            return float(self.asks.peekitem(0)[1])
        return 0.0

    def best_bid_with_qty(self) -> Tuple[Optional[float], float]:
        self._expire()
        if self.bids:
            tick, qty = self.bids.peekitem(-1)
            price = self._to_price(tick)
            self.prev_best_bid = price
            return price, float(qty)
        return self.prev_best_bid, 0.0

    def best_ask_with_qty(self) -> Tuple[Optional[float], float]:
        self._expire()
        if self.asks:
            tick, qty = self.asks.peekitem(0)
            price = self._to_price(tick)
            self.prev_best_ask = price
            return price, float(qty)
        return self.prev_best_ask, 0.0

    def spread(self) -> Optional[float]:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is not None and ba is not None:
            return ba - bb
        return self.prev_best_ask - self.prev_best_bid

    def best_bid_ex_agent(self) -> Optional[float]:
        """Best bid price considering only levels with anonymous volume."""
        self._expire()
        for tick, _total_qty in reversed(self.bids.items()):
            if self._anon_bid.get(tick, 0.0) > 0:
                price = self._to_price(tick)
                self.prev_best_bid_ex_agent = price
                return price
        return self.prev_best_bid_ex_agent

    def best_ask_ex_agent(self) -> Optional[float]:
        """Best ask price considering only levels with anonymous volume."""
        self._expire()
        for tick, _total_qty in self.asks.items():
            if self._anon_ask.get(tick, 0.0) > 0:
                price = self._to_price(tick)
                self.prev_best_ask_ex_agent = price
                return price
        return self.prev_best_ask_ex_agent

    def top_n_bid_anon(self, n: int) -> List[Tuple[float, float]]:
        """Return up to n (price, anon_qty) tuples for the best bid levels with anonymous volume, best first."""
        self._expire()
        result: List[Tuple[float, float]] = []
        for tick, _total_qty in reversed(self.bids.items()):
            anon_qty = self._anon_bid.get(tick, 0.0)
            if anon_qty > 0:
                result.append((self._to_price(tick), anon_qty))
                if len(result) == n:
                    break
        return result

    def top_n_ask_anon(self, n: int) -> List[Tuple[float, float]]:
        """Return up to n (price, anon_qty) tuples for the best ask levels with anonymous volume, best first."""
        self._expire()
        result: List[Tuple[float, float]] = []
        for tick, _total_qty in self.asks.items():
            anon_qty = self._anon_ask.get(tick, 0.0)
            if anon_qty > 0:
                result.append((self._to_price(tick), anon_qty))
                if len(result) == n:
                    break
        return result

    def spread_ex_agent(self) -> Optional[float]:
        """Spread excluding agent-only levels."""
        bb = self.best_bid_ex_agent()
        ba = self.best_ask_ex_agent()
        if bb is not None and ba is not None:
            return ba - bb
        return self.prev_best_ask_ex_agent - self.prev_best_bid_ex_agent

    def mid_price(self) -> Optional[float]:
        self._expire()
        if self.bids and self.asks:
            bb = self._to_price(self.bids.peekitem(-1)[0])
            ba = self._to_price(self.asks.peekitem(0)[0])
            return 0.5 * (bb + ba)
        else:
            return 0.5 * (self.prev_best_bid + self.prev_best_ask)
        

    def book_pressure(self, side, depth) -> Optional[float]:
        """VWAP across the top ``depth`` levels of the requested side."""
        self._expire()
        if side == "Asks":
            book = self.asks
            if len(book) < depth:
                return None
            it = book.items()
        else:
            book = self.bids
            if len(book) < depth:
                return None
            it = reversed(book.items())

        prices: List[float] = []
        weights: List[float] = []
        for tick, qty in it:
            if qty > 0:
                prices.append(self._to_price(tick))
                weights.append(qty)
            if len(prices) >= depth:
                break

        if len(prices) < depth:
            return None
        return float(np.average(prices, weights=weights))

    # ---------------------------------------------------- priority / lifetime

    def _compute_priority_index(self, side: str, tick: int, qty: float) -> float:
        """Fraction of same-side volume with priority over the incoming order.

        Matches market_simulation.OrderBook semantics: same-level volume counts
        as ahead (in the original price-time book it has time priority; here we
        keep identical semantics so the fitted lifetime sampler stays valid).
        """
        if side == "B":
            book = self.bids
            total = self._side_total_bid
            vol_ahead = 0.0
            # Iterate from best (highest) downward; bids "ahead" are price >= tick
            # (same-level volume counts as ahead, matching the ask branch and the
            # docstring contract — was previously `<=` which excluded same level).
            for level_tick, level_qty in reversed(book.items()):
                if level_tick < tick:
                    break
                vol_ahead += level_qty
        else:
            book = self.asks
            total = self._side_total_ask
            vol_ahead = 0.0
            for level_tick, level_qty in book.items():
                if level_tick > tick:
                    break
                vol_ahead += level_qty

        denom = total + qty
        return vol_ahead / denom if denom > 0 else 0.0

    def sample_lifetime(self, side: str, priority_index: float) -> float:
        if self._lifetime_sampler is not None:
            
            return self._lifetime_sampler.sample(
                priority_index, rng=self._np_rng, time=self.time
            )
            
        return self.rng.expovariate(1.0 / self.mean_order_lifespan_hours)

    def _sample_pi_zero_lifetime(self) -> float:
        """Lognormal fallback used by market_simulation when pi == 0 (verbatim)."""
        if self.time < 7:
            # mu=-6.4872, sigma^2=2.0000
            return 0.8 * float(np.exp(self.rng.normalvariate(mu=-6.4872, sigma=np.sqrt(2.0000))))
        # mu=-7.2205, sigma^2=1.9334
        return 0.5 * float(np.exp(self.rng.normalvariate(mu=-7.2205, sigma=np.sqrt(1.9334))))
    


    # ----------------------------------------------------- order entry API

    def add_market(self, side: str, qty: float, t: float) -> List[Fill]:
        if side not in ("B", "S"):
            raise ValueError("side must be 'B' or 'S'")
        if qty <= 0:
            return []
        self._sync_to_event_time(t)
        fills = self._match(taker_side=side, limit_tick=None, qty=float(qty))
        self._close_if_needed()
        return fills

    def add_limit(
        self,
        side: str,
        price: float,
        qty: float,
        t: float,
        agent: bool = False,
    ) -> Tuple[int, List[Fill]]:
        if side not in ("B", "S"):
            raise ValueError("side must be 'B' or 'S'")
        if qty <= 0:
            raise ValueError("qty must be positive")

        self._sync_to_event_time(t)

        tick = self._to_tick(price)
        order_id = self._next_id
        self._next_id += 1

        fills = self._match(taker_side=side, limit_tick=tick, qty=float(qty))
        filled = sum(f.qty for f in fills)
        remaining = float(qty) - filled

        if remaining > 1e-12 and self.time < self.end_time:
            pi = self._compute_priority_index(side, tick, remaining)
            if pi == 0.0:
                
                expiry_time = self.time + self._sample_pi_zero_lifetime()
            else:
                expiry_time = self.time + (1.2 if not self._dampener else 1.5) * self.sample_lifetime(side, pi)

            if agent:
                order = RestingOrder(
                    order_id=order_id,
                    side=side,
                    price=self._to_price(tick),
                    tick=tick,
                    qty=remaining,
                    expiry_time=expiry_time,
                    active=True,
                )
                self.agent_orders[order_id] = order
                registry = (
                    self._agent_bid_at_level if side == "B" else self._agent_ask_at_level
                )
                level = registry.get(tick)
                if level is None:
                    level = OrderedDict()
                    registry[tick] = level
                level[order_id] = order
                self._inc_level(side, tick, remaining)
                heapq.heappush(self._agent_expiry, (expiry_time, order_id))
            else:
                if side == "B":
                    self._anon_bid[tick] = self._anon_bid.get(tick, 0.0) + remaining
                else:
                    self._anon_ask[tick] = self._anon_ask.get(tick, 0.0) + remaining
                self._inc_level(side, tick, remaining)
                heapq.heappush(self._anon_expiry, (expiry_time, tick, remaining, side))

        self._close_if_needed()
        return order_id, fills

    def cancel_order(self, order_id: int) -> bool:
        order = self.agent_orders.get(order_id)
        if order is None or not order.active or order.qty <= 0:
            return False
        self._remove_agent_order(order)
        return True

    # ---------------------------------------------------------- matching

    def _match(
        self,
        taker_side: str,
        limit_tick: Optional[int],
        qty: float,
    ) -> List[Fill]:
        """Walk the maker side best-first; pro-rata fill between agent and anonymous at each level."""
        if taker_side not in ("B", "S"):
            raise ValueError("side must be 'B' or 'S'")
        if qty <= 0:
            return []

        fills: List[Fill] = []

        if taker_side == "B":
            maker_book = self.asks
            anon_map = self._anon_ask
            agent_registry = self._agent_ask_at_level
            maker_side_str = "S"
        else:
            maker_book = self.bids
            anon_map = self._anon_bid
            agent_registry = self._agent_bid_at_level
            maker_side_str = "B"

        while qty > 1e-12 and maker_book:
            if taker_side == "B":
                best_tick = maker_book.peekitem(0)[0]
                if limit_tick is not None and best_tick > limit_tick:
                    break
            else:
                best_tick = maker_book.peekitem(-1)[0]
                if limit_tick is not None and best_tick < limit_tick:
                    break

            best_price = self._to_price(best_tick)

            # --- Pro-rata split between agent and anonymous volume ---
            total_at_level = maker_book.get(best_tick, 0.0)
            if total_at_level <= 0:
                maker_book.pop(best_tick, None)
                continue

            anon_at_level = anon_map.get(best_tick, 0.0)
            level = agent_registry.get(best_tick)
            agent_at_level = 0.0
            if level:
                agent_at_level = sum(
                    o.qty for o in level.values() if o.active and o.qty > 0
                )

            trade_at_level = min(qty, total_at_level)

            # Agent share: agent_vol / (agent_vol + anon_vol)
            if agent_at_level > 0 and total_at_level > 0:
                agent_share = trade_at_level * (agent_at_level / total_at_level)
            else:
                agent_share = 0.0
            anon_share = trade_at_level - agent_share

            # 1) Fill agent orders with their pro-rata share (FIFO within agents).
            remaining_agent = agent_share
            if level and remaining_agent > 1e-12:
                for oid in list(level.keys()):
                    if remaining_agent <= 1e-12:
                        break
                    order = level[oid]
                    if not order.active or order.qty <= 0:
                        level.pop(oid, None)
                        continue
                    trade_qty = min(remaining_agent, order.qty)
                    fills.append(
                        Fill(
                            taker_side=taker_side,
                            maker_order_id=oid,
                            price=best_price,
                            qty=trade_qty,
                        )
                    )
                    remaining_agent -= trade_qty
                    order.qty -= trade_qty
                    self._dec_level(maker_side_str, best_tick, trade_qty)
                    if order.qty <= 1e-12:
                        order.active = False
                        order.qty = 0
                        level.pop(oid, None)
                if not level:
                    agent_registry.pop(best_tick, None)

            # 2) Fill anonymous volume with its pro-rata share.
            if anon_share > 1e-12:
                cur_anon = anon_map.get(best_tick, 0.0)
                if cur_anon > 0:
                    trade_qty = min(anon_share, cur_anon)
                    fills.append(
                        Fill(
                            taker_side=taker_side,
                            maker_order_id=ANON_MAKER_ID,
                            price=best_price,
                            qty=trade_qty,
                        )
                    )
                    new_anon = cur_anon - trade_qty
                    if new_anon <= 1e-12:
                        anon_map.pop(best_tick, None)
                    else:
                        anon_map[best_tick] = new_anon
                    self._dec_level(maker_side_str, best_tick, trade_qty)

            qty -= trade_at_level

            # If the level is now empty (volume drained), the SortedDict entry
            # will already have been removed by _dec_level. Otherwise we ran
            # out of incoming qty and stop.
            if best_tick in maker_book and qty <= 1e-12:
                break
            if best_tick in maker_book and maker_book[best_tick] <= 0:
                # Defensive: should not happen because _dec_level pops empties.
                maker_book.pop(best_tick, None)

        return fills


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test / micro-benchmark
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time as _time

    print("=== Basic smoke test (mirrors market_simulation.py) ===")
    # Use a long mean lifespan and disable the pi==0 lognormal fallback so
    # anon orders survive long enough for the market order to hit them.
    ob = Book(
        lifetime_sampler_config=None,
        mean_order_lifespan_hours=1e6,
        prev_best_bid=99.0,
        prev_best_ask=101.0,
        seed=0,
    )
    ob._sample_pi_zero_lifetime = lambda: 1e6  # type: ignore[assignment]

    events = [
        {"type": "limit", "side": "S", "price": 101.0, "qty": 5, "t": 0.1},
        {"type": "limit", "side": "S", "price": 103.0, "qty": 5, "t": 0.1},
        {"type": "limit", "side": "B", "price": 96.0, "qty": 10, "t": 0.1},
        {"type": "limit", "side": "S", "price": 102.0, "qty": 3, "t": 0.2},
        {"type": "limit", "side": "B", "price": 100.0, "qty": 4, "t": 0.3},
        {"type": "market", "side": "B", "qty": 6, "t": 0.4},  # hits 101 then 102
        {"type": "limit", "side": "B", "price": 101.0, "qty": 2, "t": 0.5},  # rests (best ask now 102)
    ]
    for e in events:
        if e["type"] == "market":
            f = ob.add_market(e["side"], e["qty"], e["t"])
            print("MARKET", e, "fills:", f)
        else:
            oid, f = ob.add_limit(e["side"], e["price"], e["qty"], e["t"])
            print("LIMIT", e, "id:", oid, "fills:", f)
        print("  best_bid:", ob.best_bid(), "best_ask:", ob.best_ask())

    # --- Anon expiry ---
    print("\n=== Anon expiry ===")
    ob2 = Book(lifetime_sampler_config=None, prev_best_bid=99.0, prev_best_ask=101.0, seed=1)
    ob2.add_limit("B", 100.0, 5.0, 0.0)
    print("  before expiry, best_bid_qty:", ob2.best_bid_qty())
    # Force expiry by jumping past the heap top.
    if ob2._anon_expiry:
        next_t = ob2._anon_expiry[0][0] + 1e-6
        ob2.tick(min(next_t, ob2.end_time - 1e-9))
    print("  after  expiry, best_bid_qty:", ob2.best_bid_qty())
    assert ob2.best_bid_qty() == 0.0, "anon volume should have expired"

    # --- Pro-rata matching ---
    print("\n=== Pro-rata matching ===")
    # All events at t=0.0 so nothing can expire (expiry requires t_exp <= time).
    ob3 = Book(lifetime_sampler_config=None, prev_best_bid=99.0, prev_best_ask=101.0, seed=2)
    ob3.add_limit("B", 100.0, 10.0, 0.0, agent=False)         # anon
    aid, _ = ob3.add_limit("B", 100.0, 3.0, 0.0, agent=True)  # agent
    fills = ob3.add_market("S", 5.0, 0.0)
    print("  fills:", fills)
    agent_filled = sum(f.qty for f in fills if f.maker_order_id == aid)
    anon_filled = sum(f.qty for f in fills if f.maker_order_id == ANON_MAKER_ID)
    expected_agent = 5.0 * 3.0 / 13.0
    expected_anon = 5.0 * 10.0 / 13.0
    print(f"  agent_filled={agent_filled:.4f} (expect {expected_agent:.4f}), "
          f"anon_filled={anon_filled:.4f} (expect {expected_anon:.4f})")
    assert abs(agent_filled - expected_agent) < 1e-9
    assert abs(anon_filled - expected_anon) < 1e-9

    # --- Cancel ---
    print("\n=== Cancel agent order ===")
    ob4 = Book(lifetime_sampler_config=None, prev_best_bid=99.0, prev_best_ask=101.0, seed=3)
    aid, _ = ob4.add_limit("B", 100.0, 7.0, 0.0, agent=True)
    qty_before = ob4.best_bid_qty()
    ok = ob4.cancel_order(aid)
    qty_after = ob4.best_bid_qty()
    print(f"  cancel ok={ok}, qty {qty_before} -> {qty_after}, active={ob4.agent_orders[aid].active}")
    assert ok and qty_after == 0.0 and ob4.agent_orders[aid].active is False

    # --- Priority index ---
    print("\n=== Priority index ===")
    ob5 = Book(lifetime_sampler_config=None, prev_best_bid=99.0, prev_best_ask=101.0, seed=4)
    ob5.add_limit("B", 100.0, 10.0, 0.0)
    pi_same = ob5._compute_priority_index("B", ob5._to_tick(100.0), 5.0)
    pi_better = ob5._compute_priority_index("B", ob5._to_tick(101.0), 5.0)
    print(f"  pi(same level, qty=5) = {pi_same:.4f} (expect {10/15:.4f})")
    print(f"  pi(better level)      = {pi_better:.4f} (expect 0.0)")
    assert abs(pi_same - 10 / 15) < 1e-9
    assert pi_better == 0.0

    # --- Micro-benchmark ---
    print("\n=== Micro-benchmark: 100k anon insert + expire ===")
    ob6 = Book(lifetime_sampler_config=None, prev_best_bid=99.0, prev_best_ask=101.0, seed=5)
    N = 100_000
    rng = np.random.default_rng(0)
    sides = rng.choice(["B", "S"], size=N)
    prices = rng.normal(100.0, 0.5, size=N)
    qtys = rng.uniform(0.1, 5.0, size=N)
    times = np.sort(rng.uniform(0.0, 1.0, size=N))
    t0 = _time.perf_counter()
    for i in range(N):
        ob6.add_limit(str(sides[i]), float(prices[i]), float(qtys[i]), float(times[i]))
    # Drain expiries to end of book life.
    ob6.tick(min(times[-1] + 5.0, ob6.end_time - 1e-9))
    elapsed = _time.perf_counter() - t0
    print(f"  {N} ops in {elapsed:.3f}s  =>  {N/elapsed:,.0f} ops/sec")

    print("\nAll smoke tests passed.")