%%writefile /content/drive/MyDrive/AI_stock_project/backtester.py
import pandas as pd

class Backtester:
    def __init__(self, df: pd.DataFrame, initial_capital: float = 10000.0):  
        self.df = df.copy()
        self.initial_capital = initial_capital

    def simulate_trades(self) -> pd.DataFrame:
        
        self.df["portfolio_value"] = float(self.initial_capital)  
        self.df["position"] = 0  

        
        for i in range(1, len(self.df)):
            if self.df.loc[i-1, "trend"] == 1:
                self.df.loc[i, "position"] = 1
                self.df.loc[i, "portfolio_value"] = (
                    self.df.loc[i-1, "portfolio_value"] * (1 + self.df.loc[i, "daily_return"])
                )
            else:
                self.df.loc[i, "position"] = 0
                self.df.loc[i, "portfolio_value"] = self.df.loc[i-1, "portfolio_value"]

        return self.df