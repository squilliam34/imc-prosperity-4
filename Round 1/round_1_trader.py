import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

import numpy as np

# Hard code product names to avoid typos and having to type the full name
ROOTS = 'INTARIAN_PEPPER_ROOT'
OSMIUM = 'ASH_COATED_OSMIUM'

PRODUCTS = [
    ROOTS,
    OSMIUM
]

DEFAULT_PRICES = {
    # This is the price of the roots at the end of Day 0 
    ROOTS: 13000,
    OSMIUM: 10000
}

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
        for symbol, order_depths in order_depths.items():
            compressed[symbol] = [order_depths, order_depths]

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


class Trader:

    def __init__(self):
        self.position_limits = {
            OSMIUM: 5,
            ROOTS: 5
        }

        self.ema = dict()
        for product in PRODUCTS:
            self.ema[product] = DEFAULT_PRICES[product]

        self.alpha = 0.5

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
        # Return None if one of their values == 0 due to a lack of orders
        mid = (best_bid + best_ask)/2
        if mid < 9000:
            return None
        return mid
    
    def calculate_ema(self, state: TradingState):
        """
        Calculates the exponential moving average for products
        """
        for product in PRODUCTS:
            mid = self.get_mid_price(product, state)

            if mid is None:
                continue

            else:
                self.ema[product] = self.alpha*mid + (1 - self.alpha)*self.ema[product]
    
    def trade_osmium(self, state: TradingState):
        """
        Strategy for trading osmium. FV asset so trade around the FV
        """
        position = self.get_position(OSMIUM, state)
        mu = self.ema[OSMIUM]
        eps = 8

        orders = []
        
        # How much we shift our price per unit of inventory
        skew_factor = 0.1
    
        orders = []

        skewed_mu = mu - (position * skew_factor)

        buy_price = round(skewed_mu - (eps -1))
        sell_price = round(skewed_mu + eps)

        buy_qty = self.position_limits[OSMIUM] - position
        sell_qty = -self.position_limits[OSMIUM] - position 

        buy_qty = self.position_limits[OSMIUM] - position
        sell_qty = -self.position_limits[OSMIUM] - position

        orders.append(Order(OSMIUM, buy_price, buy_qty))
        orders.append(Order(OSMIUM, sell_price, sell_qty))

        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        result = {}
        conversions = 0
        trader_data = ""

        self.calculate_ema(state)
        result[OSMIUM] = self.trade_osmium(state)

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data