# risk_management.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class PositionSizing:
    """
    Calculates position sizes based on various risk management rules
    """
    max_portfolio_risk: float = 0.02  # Max 2% of portfolio risk per trade
    max_position_size: float = 0.2  # Max 20% of portfolio in single position
    volatility_lookback: int = 20  # Days for volatility calculation
    
    def calculate_size(self, portfolio_value: float, entry_price: float, 
                      stop_loss: float, volatility: Optional[float] = None) -> int:
        """
        Calculate position size based on risk parameters
        
        Args:
            portfolio_value: Total portfolio value
            entry_price: Entry price of the asset
            stop_loss: Stop loss price
            volatility: Optional volatility measure (ATR or standard deviation)
            
        Returns:
            Number of shares to purchase
        """
        # Calculate dollar risk per share
        risk_per_share = entry_price - stop_loss
        
        # Calculate maximum dollar risk for this trade
        max_risk_amount = portfolio_value * self.max_portfolio_risk
        
        # Basic position sizing
        position_size = max_risk_amount / risk_per_share
        
        # Apply volatility adjustment if provided
        if volatility is not None:
            volatility_adjusted_size = position_size * (1 / (volatility + 0.01))
            position_size = min(position_size, volatility_adjusted_size)
        
        # Convert to shares (integer)
        shares = int(position_size)
        
        # Apply maximum position size constraint
        max_shares = int((portfolio_value * self.max_position_size) / entry_price)
        return min(shares, max_shares)

class RiskAssessor:
    """
    Assesses risk at portfolio and position levels
    """
    def __init__(self):
        self.sector_limits = {
            'max_sector_exposure': 0.3,  # Max 30% in any single sector
            'max_market_cap_exposure': 0.4  # Max 40% in small caps
        }
    
    def assess_portfolio_risk(self, portfolio: Dict[str, float], 
                             ticker_info: Dict[str, Dict]) -> Dict:
        """
        Assess portfolio-level risk metrics
        
        Args:
            portfolio: Dictionary of {ticker: market_value}
            ticker_info: Dictionary of {ticker: {'sector': ..., 'market_cap': ...}}
            
        Returns:
            Dictionary of risk metrics
        """
        total_value = sum(portfolio.values())
        if total_value == 0:
            return {}
            
        # Calculate sector exposures
        sector_exposures = {}
        for ticker, value in portfolio.items():
            sector = ticker_info.get(ticker, {}).get('sector', 'Unknown')
            sector_exposures[sector] = sector_exposures.get(sector, 0) + value
        
        # Calculate market cap exposures
        market_cap_exposures = {'small': 0, 'mid': 0, 'large': 0}
        for ticker, value in portfolio.items():
            market_cap = ticker_info.get(ticker, {}).get('market_cap', 'large')
            market_cap_exposures[market_cap] += value
        
        return {
            'sector_exposures': {k: v/total_value for k, v in sector_exposures.items()},
            'market_cap_exposures': {k: v/total_value for k, v in market_cap_exposures.items()},
            'concentration_risk': self._calculate_concentration_risk(portfolio),
            'liquidity_risk': self._calculate_liquidity_risk(portfolio, ticker_info)
        }
    
    def _calculate_concentration_risk(self, portfolio: Dict[str, float]) -> float:
        """Calculate Herfindahl-Hirschman Index for portfolio concentration"""
        total_value = sum(portfolio.values())
        if total_value == 0:
            return 0
        return sum((v/total_value)**2 for v in portfolio.values())
    
    def _calculate_liquidity_risk(self, portfolio: Dict[str, float], 
                                ticker_info: Dict[str, Dict]) -> float:
        """Calculate liquidity risk score (0-1)"""
        # This would be more sophisticated in production
        total_value = sum(portfolio.values())
        if total_value == 0:
            return 0
            
        liquidity_score = 0
        for ticker, value in portfolio.items():
            avg_volume = ticker_info.get(ticker, {}).get('avg_volume', 0)
            liquidity_score += (value/total_value) * min(1, avg_volume/1e6)  # Normalize
        
        return 1 - liquidity_score  # Higher is riskier

class StopLossCalculator:
    """
    Calculates various types of stop losses
    """
    @staticmethod
    def fixed_percentage(price: float, percentage: float) -> float:
        """Fixed percentage stop loss"""
        return price * (1 - percentage)
    
    @staticmethod
    def atr_based(price: float, atr: float, multiplier: float = 2) -> float:
        """ATR-based stop loss"""
        return price - (atr * multiplier)
    
    @staticmethod
    def volatility_based(price: float, volatility: float, 
                        risk_multiple: float = 1.5) -> float:
        """Volatility-based stop loss"""
        return price * (1 - (risk_multiple * volatility))
    
    @staticmethod
    def trailing(price: float, highest_price: float, 
                percentage: float = 0.05) -> float:
        """Trailing stop loss"""
        return highest_price * (1 - percentage)

# Example usage:
if __name__ == "__main__":
    # Position sizing example
    sizing = PositionSizing()
    shares = sizing.calculate_size(
        portfolio_value=100000,
        entry_price=50,
        stop_loss=45,
        volatility=0.1
    )
    print(f"Recommended shares to buy: {shares}")
    
    # Risk assessment example
    risk = RiskAssessor()
    portfolio = {'AAPL': 30000, 'MSFT': 20000, 'TSLA': 5000}
    ticker_info = {
        'AAPL': {'sector': 'Technology', 'market_cap': 'large', 'avg_volume': 1e7},
        'MSFT': {'sector': 'Technology', 'market_cap': 'large', 'avg_volume': 8e6},
        'TSLA': {'sector': 'Automotive', 'market_cap': 'large', 'avg_volume': 3e7}
    }
    print("\nPortfolio risk assessment:")
    print(risk.assess_portfolio_risk(portfolio, ticker_info))