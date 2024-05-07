import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf 
import statsmodels.api as sm

class strat_evaluator:
    """
    A class for evaluating trading strategies based on asset price data, trading signals,
    and fundamental series. It allows for the visualization of trading signals on asset price series,
    computation of equity curves, and calculation of various portfolio metrics.
    """
    
    def __init__(self, data, series, price='Close'):
        """
        Initialize the strat_evaluator with data and configurations.
        
        Args:
            data (pd.DataFrame): The dataset containing asset prices and trading signals.
            series (str): The column name of the fundamental series used for analysis.
            price (str): The column name representing the asset prices. Defaults to 'Close'.
        """
        data = data.copy()
        data['returns'] = np.log(data[price]).diff()
        data["strat returns"] = (data["signal"].shift() * data.returns)
        data["cumulative_returns"] = np.exp(data['strat returns'].cumsum()) - 1
        data["cumulative_returns"] = data["cumulative_returns"].fillna(0)
        data['change'] = (data["signal"].shift() != data["signal"]) * data["signal"]
        data["success"] = ((data[data.change != 0]["cumulative_returns"].diff() > 0) * 1).shift(-1)
        data['exits'] = ((data['signal'].shift() != 0) & (data['signal'] == 0)) * 1
        data.loc[data.index[0], "exits"] = 0
        
        self.data = data
        self.predictor = series
        self.mreturn = np.exp(self.data[["strat returns"]].resample('M').sum())-1
            
    def plot_signals(self, fundamental=None, title=None, ylabel=None, xlabel='Date', hsize=11, vsize=5, bounds=False):
        """
        Plot trading signals over time with the option to include fundamental data.
        
        Args:
            fundamental (str): The fundamental data series to plot. If None, uses the series provided during initialization.
            title (str): Title of the plot.
            ylabel (str): Y-axis label.
            xlabel (str): X-axis label, defaults to 'Date'.
            hsize (int): Horizontal size of the plot.
            vsize (int): Vertical size of the plot.
            bounds (bool): If True, will plot additional boundary lines
        """
        if fundamental is None:
            fundamental = self.predictor
        if title is None:
            title = "Signals v. " + fundamental
        if ylabel is None:
            ylabel = fundamental
            
        df_copy = self.data
        fig, ax = plt.subplots(figsize=(hsize, vsize))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.plot(df_copy[fundamental], label=fundamental)
        
        # Different scenarios for success and failure in trades
        longsuccess = df_copy[(df_copy["success"] == 1) & (df_copy["signal"] == 1)]
        longfail = df_copy[(df_copy["success"] == 0) & (df_copy["signal"] == 1)]
        shortsuccess = df_copy[(df_copy["success"] == 1) & (df_copy["signal"] == -1)]
        shortfail = df_copy[(df_copy["success"] == 0) & (df_copy["signal"] == -1)]
        
        ax.scatter(longsuccess.index, longsuccess[fundamental], color='blue', label='Long Success', s=100)
        ax.scatter(longfail.index, longfail[fundamental], color='blue', label='Long Failure', marker='x', s=100)
        ax.scatter(shortsuccess.index, shortsuccess[fundamental], color='red', label='Short Success', s=100)
        ax.scatter(shortfail.index, shortfail[fundamental], color='red', label='Short Failure', marker='x', s=100)
        
        # Exit points
        if df_copy['exits'].abs().sum() != 0:
            ax.scatter(df_copy[df_copy.exits == 1].index, df_copy[df_copy.exits == 1][fundamental], color='black', label='Exit Point', s=30)
        
        if bounds:
            ax.fill_between(df_copy.index, df_copy.Lower, df_copy.Upper, color='gray', alpha=0.3, label='Boundaries')
        
        ax.legend()
        ax.grid(True)
        plt.show()
    
    def plot_equity(self, title='Strategy Returns', ylabel='Returns (%)', xlabel='Date', hsize=11, vsize=5):
        """
        Plot the equity curve of the trading strategy.
        
        Args:
            title (str): Title of the plot.
            ylabel (str): Y-axis label for returns.
            xlabel (str): X-axis label for date.
            hsize (int): Horizontal size of the plot.
            vsize (int): Vertical size of the plot.
        """
        plt.figure(figsize=(hsize, vsize))
        plt.title(title)
        plt.plot(self.data['cumulative_returns']*100, label='Cumulative Returns')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def calcArAvgReturn(self, returns=None, annualize=False, periods_in_year=12):
        """
        Calculate the average return of a portfolio, with the option to annualize.
        
        Args:
            returns (np.array): Specific returns to calculate the average. If None, uses monthly returns stored in the object.
            annualize (bool): If True, annualizes the return based on the number of periods per year.
            periods_in_year (int): The number of periods in a year, used for annualization.
        
        Returns:
            float: The average return of the portfolio.
        """
        if returns is None:
            rets = self.mreturn
        else:
            rets = returns
        
        mean_m_return = np.mean(rets)
        if annualize:
            mean_ret = periods_in_year * mean_m_return
            return max(mean_ret, -1.0)
        else:
            return mean_m_return
        
    def calcSD(self, returns=None, annualize=False) -> float:
        """
        Calculate the standard deviation of a vector of simple returns with the option to annualize.
        
        Args:
            returns (np.array): Vector of simple returns at any frequency. If None, uses the returns stored in the object.
            annualize (bool): If True, annualizes the standard deviation based on monthly returns.
        
        Returns:
            float: The calculated standard deviation. Annualized if specified.
        """
        if returns is None:
            rets = self.mreturn
        else:
            rets = returns

        sd_m = np.std(rets)
        return np.sqrt(12) * sd_m if annualize else sd_m

    def calcSharpe(self, risk_free_returns=None) -> float:
        """
        Calculate the annual Sharpe Ratio of a vector of simple returns.
        
        Args:
            risk_free_returns (np.array): Vector of simple returns of the risk-free rate. If provided, it must match the size of the portfolio returns.
        
        Returns:
            float: The calculated annualized Sharpe ratio.
        """
        if risk_free_returns is not None:
            if not isinstance(risk_free_returns, np.ndarray):
                raise TypeError("Input 'risk_free_returns' must be a NumPy array")
            if np.isnan(risk_free_returns).any():
                raise ValueError("Input 'risk_free_returns' contains NaN values")
            if risk_free_returns.size != self.mreturn.values.size:
                raise ValueError("'returns' and 'risk_free_returns' must be of the same size")

            returns = self.mreturn.values.flatten() - risk_free_returns
        else:
            returns = self.mreturn.values.flatten()

        return self.calcArAvgReturn(returns, annualize=True) / self.calcSD(returns, annualize=True)

    def calcSortino(self, risk_free_returns=None, target_return=0.0) -> float:
        """
        Calculate the annual Sortino Ratio of a vector of simple returns, focusing only on the downside deviation.
        
        Args:
            risk_free_returns (np.array): Vector of simple returns of the risk-free rate, used to adjust returns for the risk-free rate.
            target_return (float): The target return level below which the deviation is considered downside.
        
        Returns:
            float: The calculated annualized Sortino ratio.
        """
        if risk_free_returns is not None:
            if not isinstance(risk_free_returns, np.ndarray):
                raise TypeError("Input 'risk_free_returns' must be a NumPy array")
            if np.isnan(risk_free_returns).any():
                raise ValueError("Input 'risk_free_returns' contains NaN values")
            if risk_free_returns.size != self.mreturn.size:
                raise ValueError("'returns' and 'risk_free_returns' must be of the same size")

            returns = self.mreturn.values.flatten() - risk_free_returns
        else:
            returns = self.mreturn.values.flatten()

        downside_returns = np.copy(returns)
        downside_returns[returns > target_return] = 0

        return self.calcArAvgReturn(returns, annualize=True) / self.calcSD(downside_returns, annualize=True)

    def calcMaxDrawdown(self) -> float:
        """
        Calculate the maximum drawdown for the portfolio returns.
        
        Returns:
            float: The maximum drawdown in simple return units over the calculated period.
        """
        cumulative_ret = np.cumprod(1 + self.mreturn)
        roll_max = np.maximum.accumulate(cumulative_ret)
        drawdowns = cumulative_ret / roll_max - 1
        return np.min(drawdowns)

    def calcPortfolioStatistics(self, code='^GSPC', risk_free_returns=None) -> dict:
        """
        Calculate alpha and beta of the portfolio against a market index.
        
        Args:
            code (str): The market index code to compare against, defaults to '^GSPC' for S&P 500.
            risk_free_returns (np.array): Vector of simple returns of the risk-free rate. Used if provided.
        
        Returns:
            dict: A dictionary containing 'alpha' and 'beta' of the portfolio relative to the market index.
        """
        market_returns = yf.download(code)['Adj Close'].pct_change()
        market_returns = (market_returns + 1).resample('M').prod() - 1

        if risk_free_returns is not None:
            if risk_free_returns.size != self.mreturn.size:
                raise ValueError("'returns' and 'risk_free_returns' must be of the same size")

            returns = self.mreturn.values.flatten() - risk_free_returns
            market_returns = market_returns.loc[self.mreturn.index]
            market_returns = market_returns.values - risk_free_returns
        else:
            returns = self.mreturn.values.flatten()
            market_returns = market_returns.loc[self.mreturn.index].values

        X = sm.add_constant(market_returns * 100)
        model = sm.OLS(returns * 100, X)
        results = model.fit()
        
        # Return alphas betas
        return {'alpha': results.params[0], 'beta': results.params[1]}
