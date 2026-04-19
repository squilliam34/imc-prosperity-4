import json
from typing import Any
import numpy as np

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()

class KalmanFilter(object):
    def __init__(self, F = None, Z = None, Q = None, eps = None, P = None, x0 = None):

        if(F is None or Z is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = Z.shape[1]

        # The transition matrix, set F = 1 - like a random walk
        self.F = F

        # Observation model, set Z = 1
        self.Z = Z

        # Process noise - Hidden Pattern
        self.Q = np.eye(self.n) if Q is None else Q

        # Measurement noise
        self.eps = np.eye(self.n) if eps is None else eps

        # Error covariance
        self.P = np.eye(self.n) if P is None else P

        # state estimate
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) 
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.Z, self.x)
        S = self.eps + np.dot(self.Z, np.dot(self.P, self.Z.T))
        K = np.dot(np.dot(self.P, self.Z.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.Z), self.P), 
        	(I - np.dot(K, self.Z)).T) + np.dot(np.dot(K, self.eps), K.T)

    def get_eps(self):
        return self.eps

    def get_P(self):
        return self.P

    def get_Q(self):
        return self.Q

class Trader:
    def __init__(self):

        self.position_limits = {
            OSMIUM: 3,
            ROOTS: 5
        }

        self.price_history = []
        self.WINDOW_SIZE = 10

    # Utils 
    def get_position(self, product, state: TradingState):
        """
        Retrieves your current position for a product
        """
        return state.position.get(product, 0)

    def get_mid_price(self, product, state: TradingState):
        """
        Calculates the mid price from the bid-ask spread for a product        
        """

        market_bids = state.order_depths[product].buy_orders
        market_asks = state.order_depths[product].sell_orders

        # If the book is one-sided, return None
        if not market_bids or not market_asks:
            return None
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)

        if best_ask <= best_bid:
            return None
        # Return None if one of their values == 0 due to a lack of orders
        return (best_bid + best_ask)/2

    def calculate_microprice(self, product, state: TradingState):
        """
        Calculates the microprice of a given product
        """
        orders = state.order_depths
        market_bids = orders[product].buy_orders
        market_asks = orders[product].sell_orders
        best_ask = best_bid = ask_vol = bid_vol = 0
        if market_asks:
            best_ask = min(market_asks)
            ask_vol = abs(orders[product].sell_orders[best_ask])
        if market_bids:
            best_bid = max(market_bids)
            bid_vol = orders[product].buy_orders[best_bid]

        if bid_vol == 0 & ask_vol == 0:
            return 0

        return (best_bid*bid_vol + best_ask*ask_vol) / (bid_vol + ask_vol)

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        # TODO: Add logic

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data