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
            OSMIUM: 3,
            ROOTS: 5
        }

        self.ema = dict()
        for product in PRODUCTS:
            self.ema[product] = DEFAULT_PRICES[product]

        self.alpha = 0.5

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
    
    def trade_roots(self, state: TradingState):
        orders = []
        
        mid_price = self.get_mid_price(ROOTS, state)
        if mid_price is not None:
            self.price_history.append(mid_price)
        
        # Keep window size manageable
        if len(self.price_history) > self.WINDOW_SIZE:
            self.price_history.pop(0)
        
        # 2. Estimate the "Drift" (Simple Linear Trend)
        prices = np.array(self.price_history)
        # Change per tick over the window
        if len(prices) == 1:
            drift = 0
        else:
            drift = (prices[-1] - prices[0]) / max(len(prices) - 1, 1)

        expected_price = prices[-1] + drift
        residual = prices - expected_price

        zscore = residual[-1] / np.std(residual)

        buy_qty = 0
        sell_qty = 0
        if zscore < -0.1:
            # buy
            buy_qty = 1
        if zscore > 0.1:
            # very small sell
            sell_qty = -1

        limit = self.position_limits[ROOTS]
        position = self.get_position(ROOTS, state)
        buy_qty = min(buy_qty, limit - position)
        sell_qty = min(sell_qty, position + limit)

        orders.append(Order(ROOTS, int(expected_price-1), buy_qty))
        orders.append(Order(ROOTS, int(expected_price+3), sell_qty))

        return orders
    
    def trade_osmium(self, state: TradingState):
        """
        Strategy for trading osmium. Use microprice to try to predict
        where in the cycle osmium is going
        """
        position = self.get_position(OSMIUM, state)
        limit = self.position_limits[OSMIUM]
        micro_price = self.calculate_microprice(OSMIUM, state) if self.calculate_microprice(OSMIUM, state) != 0 else self.ema[OSMIUM]
        mid_price = self.get_mid_price(OSMIUM, state) if self.get_mid_price(OSMIUM, state) else self.ema[OSMIUM]
        orders = [] 
        order_depth = state.order_depths[OSMIUM]

        bid_price = 0
        ask_price = 0

        bids = order_depth.buy_orders
        if bids:
            bid_price = max(bids) 
        asks = order_depth.sell_orders
        if asks:
            ask_price = min(asks) 

        buy_qty = limit - position
        ask_qty = - limit - position

        if position > limit*0.5:
            buy_qty = 0
        if position < -limit*0.5:
            ask_qty = 0

        threshold = 3

        if micro_price < mid_price - threshold:
            # there should be more selling in the market
            if bid_price == 0:
                bid_price = np.floor(micro_price)
            if ask_price == 0:
                ask_price = np.ceil(micro_price + 3)
        elif micro_price > mid_price + threshold:
            # there should be more buying in the market
            if bid_price == 0:
                bid_price = np.floor(micro_price - 3)

            if ask_price == 0:
                ask_price = np.ceil(micro_price)
        else:
            if bid_price == 0:
                bid_price = np.floor(mid_price - 3)
            if ask_price == 0:
                ask_price = np.ceil(mid_price + 3)
            
        orders.append(Order(OSMIUM, int(bid_price), buy_qty))
        orders.append(Order(OSMIUM, int(ask_price), ask_qty))

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
        result[ROOTS] = self.trade_roots(state)

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data