import json
from typing import Any

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


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        for product, order_depth in state.order_depths.items():
            if product == 'TOMATOES':
                position = state.position.get("TOMATOES", 0)

                orders = []
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_vol = order_depth.buy_orders[best_bid]

                best_ask = min(order_depth.sell_orders.keys())
                best_ask_vol = abs(order_depth.sell_orders[best_ask]) # Use abs() because ask vol is negative

                mid_price = (best_bid + best_ask) / 2

                # Calculate percent of buyers out of orders
                imbalance = best_bid_vol / (best_bid_vol + best_ask_vol)

                # If percent of buyers is high, price will likely go UP. 
                # We should raise our buy price to make sure we get in, or raise our sell price to capture more profit.
                if imbalance > 0.7:
                    # Bullish: favor buying
                    orders.append(Order(product, int(mid_price), 10))     # Buy closer to mid
                    orders.append(Order(product, int(mid_price + 7), -10)) # Sell higher up
                elif imbalance < 0.3:
                    # Bearish: favor selling, noy buying at all
                    if position > 10:
                        orders.append(Order(product, int(mid_price-5), -10)) # Favor selling even more if we have a large position
                    else:
                        orders.append(Order(product, int(mid_price), -10))    # Sell closer to mid
                else:
                    # Balanced
                    orders.append(Order(product, int(mid_price - 5), 10))
                    if position > 10:
                        orders.append(Order(product, int(mid_price), -10)) # Favor selling if we have a large position
                    else:
                        orders.append(Order(product, int(mid_price + 5), -10))
                result[product] = orders
            if product == 'EMERALDS':
                position = state.position.get('EMERALDS', 0)
                orders = []

                mu = 10000
                eps = 5
                # Buy using eps window around mu - the FV of the asset

                if position > 10:
                    # Favor selling
                    orders.append(Order(product, mu, -10))
                elif position < -10:
                    # Favor buying
                    orders.append(Order(product, mu, 10))
                else:
                    orders.append(Order(product, mu + eps, -7))
                    orders.append(Order(product, mu - eps, 7))
                

                result[product] = orders


        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data