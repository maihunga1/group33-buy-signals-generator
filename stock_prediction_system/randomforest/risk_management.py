import numpy as np
import pandas as pd
from config import RISK_PARAMS


MAX_POSITION_SIZE = RISK_PARAMS['max_position_size']
RISK_PER_TRADE = RISK_PARAMS['max_portfolio_risk']
MAX_OPEN_POSITIONS = 5

class PositionSizing:
    """
    Implements position sizing strategies based on risk management principles:
    - Fixed percentage of portfolio
    - Risk-based position sizing
    - ATR-based position sizing
    - Kelly criterion (optional)
    """
    
    def __init__(self, default_risk_pct=RISK_PER_TRADE, max_position_size=MAX_POSITION_SIZE):
        """
        Initialize the position sizer with default parameters
        
        Args:
            default_risk_pct: Default percentage of portfolio to risk per trade (0-1)
            max_position_size: Maximum position size as percentage of portfolio (0-1)
        """
        self.default_risk_pct = default_risk_pct
        self.max_position_size = max_position_size
        
    def calculate_size(self, portfolio_value, entry_price, stop_loss=None, volatility=None,
                       risk_pct=None, method='risk_based'):
        """
        Calculate position size based on the specified method
        
        Args:
            portfolio_value: Current portfolio value
            entry_price: Entry price of the asset
            stop_loss: Stop loss price level
            volatility: Volatility measure (e.g., ATR)
            risk_pct: Risk percentage to use (overrides default)
            method: Position sizing method ('fixed', 'risk_based', 'atr', 'kelly')
            
        Returns:
            Number of shares/contracts to trade
        """
        risk_pct = risk_pct if risk_pct is not None else self.default_risk_pct
        
        if method == 'fixed':
            # Fixed percentage of portfolio
            position_value = portfolio_value * risk_pct
            shares = position_value / entry_price
            
        elif method == 'risk_based':
            # Risk-based position sizing using stop loss
            if stop_loss is None:
                # Default to 5% risk if no stop loss specified
                stop_loss = entry_price * 0.95
                
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share <= 0:
                return 0  # Avoid division by zero
                
            dollar_risk = portfolio_value * risk_pct
            shares = dollar_risk / risk_per_share
            
        elif method == 'atr':
            # ATR-based position sizing
            if volatility is None:
                # Fall back to risk-based sizing if no volatility provided
                return self.calculate_size(portfolio_value, entry_price, stop_loss, 
                                         None, risk_pct, 'risk_based')
                
            # Use 2x ATR as risk measure
            risk_per_share = 2 * volatility
            dollar_risk = portfolio_value * risk_pct
            shares = dollar_risk / risk_per_share
            
        elif method == 'kelly':
            # Kelly criterion (if win rate and profit factor are known)
            # This is a placeholder - in real implementation, would need historical stats
            win_rate = 0.55  # Example
            avg_win_loss_ratio = 1.5  # Example
            
            kelly_pct = win_rate - (1 - win_rate) / avg_win_loss_ratio
            kelly_pct = max(0, kelly_pct * 0.5)  # Half-Kelly for safety
            
            position_value = portfolio_value * min(kelly_pct, self.max_position_size)
            shares = position_value / entry_price
        
        else:
            raise ValueError(f"Unknown position sizing method: {method}")
        
        # Apply maximum position size constraint
        max_shares = (portfolio_value * self.max_position_size) / entry_price
        shares = min(shares, max_shares)
        
        # Return whole number of shares
        return int(shares)
    
    def adjust_for_correlation(self, base_size, correlation_matrix, open_positions):
        """
        Adjust position size based on correlation with existing positions
        
        Args:
            base_size: Base position size calculated earlier
            correlation_matrix: Matrix of asset correlations
            open_positions: Dictionary of currently open positions
            
        Returns:
            Adjusted position size
        """
        # If no open positions or no correlation data, return base size
        if not open_positions or correlation_matrix is None:
            return base_size
            
        # Calculate average correlation with existing positions
        correlations = []
        for pos_ticker in open_positions:
            if pos_ticker in correlation_matrix.index:
                for ticker in correlation_matrix.columns:
                    if ticker in open_positions:
                        correlations.append(correlation_matrix.loc[pos_ticker, ticker])
        
        if not correlations:
            return base_size
            
        avg_correlation = np.mean(correlations)
        
        # Adjust size based on correlation
        # High correlation = reduce size to avoid concentration
        correlation_factor = 1 - (avg_correlation * 0.5)  # 0.5 to dampen effect
        
        return int(base_size * max(correlation_factor, 0.5))  # Never reduce by more than 50%


class RiskAssessor:
    """
    Assesses and monitors portfolio risk levels including:
    - Portfolio volatility
    - Drawdown tracking
    - Sector/asset exposure
    - VaR calculation (optional)
    """
    
    def __init__(self, max_portfolio_risk=0.25, max_sector_exposure=0.3, 
                 max_single_position=MAX_POSITION_SIZE):
        """
        Initialize the risk assessor
        
        Args:
            max_portfolio_risk: Maximum acceptable portfolio volatility
            max_sector_exposure: Maximum exposure to any one sector
            max_single_position: Maximum single position size
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_sector_exposure = max_sector_exposure
        self.max_single_position = max_single_position
        self.current_drawdown = 0
        self.peak_value = 0
        
    def update_drawdown(self, current_value):
        """
        Update and track drawdown values
        
        Args:
            current_value: Current portfolio value
            
        Returns:
            Current drawdown percentage
        """
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
            
        return self.current_drawdown
    
    def calculate_portfolio_var(self, positions, returns_data, confidence=0.95, timeframe=1):
        """
        Calculate Value at Risk for the portfolio
        
        Args:
            positions: Dictionary of positions with ticker and value
            returns_data: DataFrame of historical returns
            confidence: Confidence level (typically 0.95 or 0.99)
            timeframe: Time horizon in days
            
        Returns:
            VaR value as portfolio percentage
        """
        if not positions or returns_data.empty:
            return 0
            
        # Extract returns for current positions
        portfolio_returns = []
        portfolio_value = sum(pos['value'] for pos in positions.values())
        
        for ticker, pos in positions.items():
            if ticker in returns_data.columns:
                weight = pos['value'] / portfolio_value
                portfolio_returns.append(returns_data[ticker] * weight)
        
        if not portfolio_returns:
            return 0
            
        # Combine weighted returns
        portfolio_return_series = pd.concat(portfolio_returns, axis=1).sum(axis=1)
        
        # Calculate VaR
        var_percentile = 1 - confidence
        var = portfolio_return_series.quantile(var_percentile)
        
        # Scale for timeframe (using square root of time rule)
        var_adjusted = var * np.sqrt(timeframe)
        
        return abs(var_adjusted)
    
    def check_sector_exposure(self, positions, sector_mapping):
        """
        Check if sector exposure exceeds limits
        
        Args:
            positions: Dictionary of positions with ticker and value
            sector_mapping: Dictionary mapping tickers to sectors
            
        Returns:
            Dictionary of sector exposures and boolean indicating if limits exceeded
        """
        if not positions or not sector_mapping:
            return {}, False
            
        portfolio_value = sum(pos['value'] for pos in positions.values())
        sector_exposure = {}
        
        for ticker, pos in positions.items():
            sector = sector_mapping.get(ticker, 'Unknown')
            if sector not in sector_exposure:
                sector_exposure[sector] = 0
            sector_exposure[sector] += pos['value'] / portfolio_value
        
        # Check if any sector exceeds maximum exposure
        exceeded = any(exposure > self.max_sector_exposure for exposure in sector_exposure.values())
        
        return sector_exposure, exceeded
    
    def can_take_new_position(self, portfolio, new_position_value, max_open_positions=MAX_OPEN_POSITIONS):
        """
        Determine if a new position can be taken based on risk constraints
        
        Args:
            portfolio: Current portfolio state
            new_position_value: Value of the new position
            max_open_positions: Maximum number of open positions allowed
            
        Returns:
            Boolean indicating if position can be taken and reason if not
        """
        # Check number of open positions
        if len(portfolio.get('positions', {})) >= max_open_positions:
            return False, "Maximum number of open positions reached"
        
        # Check position size
        portfolio_value = portfolio.get('value', 0)
        if portfolio_value <= 0:
            return False, "Invalid portfolio value"
            
        position_pct = new_position_value / portfolio_value
        if position_pct > self.max_single_position:
            return False, f"Position size ({position_pct:.1%}) exceeds maximum ({self.max_single_position:.1%})"
        
        # Check drawdown
        if self.current_drawdown > 0.2:  # 20% drawdown
            return False, f"Current drawdown ({self.current_drawdown:.1%}) exceeds threshold"
            
        return True, ""


class StopLossCalculator:
    """
    Calculates and manages different types of stop losses:
    - Fixed percentage
    - ATR-based
    - Support/resistance based
    - Trailing stops
    """
    
    def __init__(self):
        """Initialize the stop loss calculator"""
        pass
    
    def fixed_percentage(self, entry_price, risk_pct=0.05, direction='long'):
        """
        Calculate fixed percentage stop loss
        
        Args:
            entry_price: Entry price of the position
            risk_pct: Risk percentage (default 5%)
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Stop loss price
        """
        if direction == 'long':
            return entry_price * (1 - risk_pct)
        else:  # short
            return entry_price * (1 + risk_pct)
    
    def atr_based(self, entry_price, atr, multiplier=2.0, direction='long'):
        """
        Calculate ATR-based stop loss
        
        Args:
            entry_price: Entry price of the position
            atr: Average True Range value
            multiplier: ATR multiplier (typically 2-3)
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Stop loss price
        """
        if direction == 'long':
            return entry_price - (atr * multiplier)
        else:  # short
            return entry_price + (atr * multiplier)
    
    def support_resistance(self, entry_price, levels, direction='long', buffer_pct=0.01):
        """
        Set stop loss based on nearest support/resistance level
        
        Args:
            entry_price: Entry price of the position
            levels: List of support/resistance price levels
            direction: Trade direction ('long' or 'short')
            buffer_pct: Buffer percentage to add/subtract from level
            
        Returns:
            Stop loss price
        """
        if not levels:
            # Fall back to fixed percentage
            return self.fixed_percentage(entry_price, direction=direction)
            
        if direction == 'long':
            # Find nearest support below entry
            supports = [level for level in levels if level < entry_price]
            if supports:
                nearest_support = max(supports)
                return nearest_support * (1 - buffer_pct)
            else:
                # No support found, use fixed percentage
                return self.fixed_percentage(entry_price, direction=direction)
        else:  # short
            # Find nearest resistance above entry
            resistances = [level for level in levels if level > entry_price]
            if resistances:
                nearest_resistance = min(resistances)
                return nearest_resistance * (1 + buffer_pct)
            else:
                # No resistance found, use fixed percentage
                return self.fixed_percentage(entry_price, direction=direction)
    
    def trailing_stop(self, entry_price, current_price, initial_stop, trail_pct=0.05):
        """
        Calculate trailing stop loss that moves with price
        
        Args:
            entry_price: Entry price of the position
            current_price: Current market price
            initial_stop: Initial stop loss price
            trail_pct: Trailing percentage
            
        Returns:
            Updated stop loss price
        """
        if current_price > entry_price:  # Long position in profit
            trail_stop = current_price * (1 - trail_pct)
            return max(trail_stop, initial_stop)
        else:  # Short position or long position not in profit
            return initial_stop
    
    def calculate_chandelier_exit(self, high_price, atr, multiplier=3.0):
        """
        Calculate Chandelier Exit (popular trailing stop method)
        
        Args:
            high_price: Highest price since entry
            atr: Average True Range value
            multiplier: ATR multiplier
            
        Returns:
            Chandelier exit price
        """
        return high_price - (atr * multiplier)
    
    def adjust_for_volatility(self, base_stop, volatility, avg_volatility, 
                             direction='long', sensitivity=0.5):
        """
        Adjust stop loss based on current volatility relative to average
        
        Args:
            base_stop: Base stop loss price
            volatility: Current volatility (e.g., ATR)
            avg_volatility: Average volatility over lookback period
            direction: Trade direction ('long' or 'short')
            sensitivity: Adjustment sensitivity (0-1)
            
        Returns:
            Volatility-adjusted stop loss
        """
        if avg_volatility == 0:
            return base_stop
            
        # Calculate volatility ratio
        vol_ratio = volatility / avg_volatility
        
        # Adjust stop based on volatility
        adjustment = (vol_ratio - 1) * sensitivity
        
        if direction == 'long':
            # Wider stop for higher volatility
            return base_stop * (1 - adjustment)
        else:  # short
            return base_stop * (1 + adjustment)


def calculate_max_drawdown(equity_curve):
    """
    Calculate maximum drawdown from equity curve
    
    Args:
        equity_curve: Series or array of equity values
        
    Returns:
        Maximum drawdown as a percentage
    """
    if len(equity_curve) <= 1:
        return 0
        
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calculate drawdown in percentage terms
    drawdown = (running_max - equity_curve) / running_max
    
    return np.max(drawdown)


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate Sharpe ratio from returns
    
    Args:
        returns: Series or array of period returns
        risk_free_rate: Risk-free rate for the period
        
    Returns:
        Sharpe ratio
    """
    if len(returns) <= 1:
        return 0
        
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0


def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """
    Calculate Sortino ratio (only penalizes downside deviation)
    
    Args:
        returns: Series or array of period returns
        risk_free_rate: Risk-free rate for the period
        
    Returns:
        Sortino ratio
    """
    if len(returns) <= 1:
        return 0
        
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
    
    return (np.mean(excess_returns) / downside_deviation) if downside_deviation > 0 else 0