%%writefile /content/drive/MyDrive/AI_stock_project/datapreprocessor.py
import pandas as pd
import numpy as np

class DataPreprocessor:

    @staticmethod
    def predict_today_trend(df: pd.DataFrame, threshold: float = 0.5) -> str:
        """
        Predicts today's trend using the latest momentum score.

        """
        if df.empty:
          raise ValueError("DataFrame is empty after preprocessing. Check input data.")

    # Ensure momentum_score exists
        if "momentum_score" not in df.columns:
            df = DataPreprocessor.calculate_momentum_score(df)

    # Safely get the last row
        try:
          latest_score = df["momentum_score"].iloc[-1]
        except IndexError:
          raise IndexError("No data available to make a prediction.")



        # Assume the last row is "today"

        if latest_score > threshold:
            return "Bullish 📈"
        elif latest_score < -threshold:
            return "Bearish 📉"
        else:
            return "Neutral ➖"





    @staticmethod

    def clean_data(df: pd.DataFrame) -> pd.DataFrame:

      df.rename(columns={"mTIMESTAMP": "date"}, inplace=True)

      df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y', errors='coerce')

      required_cols = ["CH_CLOSING_PRICE", "CH_TRADE_HIGH_PRICE", "CH_TRADE_LOW_PRICE"]




      df.sort_values("date", inplace=True)
      return df


    @staticmethod
    def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
      df = df.copy()

      if "CH_CLOSING_PRICE" in df.columns:
        df["daily_return"] = df["CH_CLOSING_PRICE"].pct_change().astype(float)
      else:
        raise ValueError("Missing necessary closing price column for daily return calculation!")
      return df

    @staticmethod
    def add_moving_averages(df: pd.DataFrame, windows: list = [7, 14]) -> pd.DataFrame:
        """
        Adds rolling average columns for specified window sizes.
        """
        df = df.copy()
        if "CH_CLOSING_PRICE" in df.columns:
            for window in windows:
                df[f"rolling_avg_{window}"] = df["CH_CLOSING_PRICE"].rolling(window=window, min_periods=1).mean()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
      #print("Before RSI calculation, columns:", df.columns)

      if "CH_CLOSING_PRICE" not in df.columns:
        raise ValueError("Missing 'CH_CLOSING_PRICE' column in DataFrame!")

    # Compute RSI using closing price
      delta = df["CH_CLOSING_PRICE"].diff()
      gain = delta.where(delta > 0, 0)
      loss = -delta.where(delta < 0, 0)

      avg_gain = gain.rolling(window=period, min_periods=1).mean()
      avg_loss = loss.rolling(window=period, min_periods=1).mean()

      rs = avg_gain / (avg_loss + 1e-10)
      df["rsi"] = 100 - (100 / (1 + rs))

      #print("After RSI calculation, columns:", df.columns)

      return df


    @staticmethod
    def add_macd(df: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
      #print("Before MACD calculation, columns:", df.columns)

      if "CH_CLOSING_PRICE" not in df.columns:
          raise ValueError("Missing 'CH_CLOSING_PRICE' column in DataFrame!")

    # Calculate MACD
      short_ema = df["CH_CLOSING_PRICE"].ewm(span=short_window, adjust=False).mean()
      long_ema = df["CH_CLOSING_PRICE"].ewm(span=long_window, adjust=False).mean()
      df["macd"] = short_ema - long_ema
      df["macd_signal"] = df["macd"].ewm(span=signal_window, adjust=False).mean()

      #print("After MACD calculation, columns:", df.columns)

      return df


    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """
        Adds Bollinger Bands: middle, upper, and lower.
        """
        df = df.copy()
        if "CH_CLOSING_PRICE" in df.columns:
            df["bollinger_mid"] = df["CH_CLOSING_PRICE"].rolling(window=window, min_periods=1).mean()
            df["bollinger_std"] = df["CH_CLOSING_PRICE"].rolling(window=window, min_periods=1).std()
            df["bollinger_upper"] = df["bollinger_mid"] + num_std * df["bollinger_std"]
            df["bollinger_lower"] = df["bollinger_mid"] - num_std * df["bollinger_std"]
        return df

    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Adds the Average Directional Index (ADX) to measure trend strength.
        Requires 'high', 'low', and 'close' columns.
        """
        df = df.copy()
        if all(col in df.columns for col in ["CH_TRADE_HIGH_PRICE", "CH_TRADE_LOW_PRICE", "CH_CLOSING_PRICE"]):
            df["tr"] = np.maximum(df["CH_TRADE_HIGH_PRICE"] - df["CH_TRADE_LOW_PRICE"],
                                  np.maximum(abs(df["CH_TRADE_HIGH_PRICE"] - df["CH_CLOSING_PRICE"].shift(1)),
                                             abs(df["CH_TRADE_LOW_PRICE"] - df["CH_CLOSING_PRICE"].shift(1))))
            atr = df["tr"].rolling(window=period, min_periods=1).mean()
            df["up_move"] = df["CH_TRADE_HIGH_PRICE"] - df["CH_TRADE_HIGH_PRICE"].shift(1)
            df["down_move"] = df["CH_TRADE_LOW_PRICE"].shift(1) - df["CH_TRADE_LOW_PRICE"]
            df["plus_dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
            df["minus_dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)
            plus_di = 100 * (df["plus_dm"].rolling(window=period, min_periods=1).sum() / atr)
            minus_di = 100 * (df["minus_dm"].rolling(window=period, min_periods=1).sum() / atr)
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
            df["adx"] = dx.rolling(window=period, min_periods=1).mean()
            df.drop(columns=["tr", "up_move", "down_move", "plus_dm", "minus_dm"], inplace=True)
        return df

    @staticmethod
    def add_obv(df):
      #print("Before OBV calculation, columns:", df.columns)

    # Make sure 'CH_CLOSING_PRICE' and 'CH_TOT_TRADED_QTY' exist
      if "CH_CLOSING_PRICE" not in df.columns or "CH_TOT_TRADED_QTY" not in df.columns:
        raise ValueError("Missing necessary columns for OBV calculation!")

      df["obv_direction"] = np.where(df["CH_CLOSING_PRICE"].diff() > 0, 1, -1)
      df["obv"] = (df["obv_direction"] * df["CH_TOT_TRADED_QTY"]).cumsum()


      #print("After OBV calculation, columns:", df.columns)
      return df




    @staticmethod
    def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:

      df = df.copy()

    # Ensure required columns exist
      if all(col in df.columns for col in ["CH_CLOSING_PRICE", "CH_TRADE_HIGH_PRICE", "CH_TRADE_LOW_PRICE"]):
        low_min = df["CH_TRADE_LOW_PRICE"].rolling(window=k_period, min_periods=1).min()
        high_max = df["CH_TRADE_HIGH_PRICE"].rolling(window=k_period, min_periods=1).max()
        df["stoch_%K"] = ((df["CH_CLOSING_PRICE"] - low_min) / (high_max - low_min)) * 100
        df["stoch_%D"] = df["stoch_%K"].rolling(window=d_period, min_periods=1).mean()
      else:
        missing_cols = [col for col in ["CH_CLOSING_PRICE", "CH_TRADE_HIGH_PRICE", "CH_TRADE_LOW_PRICE"] if col not in df.columns]
        raise ValueError(f"Missing columns for Stochastic calculation: {missing_cols}")

      #print("After Stochastic calculation, columns:", df.columns)  # Debugging line
      return df

    @staticmethod
    def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:

      df = df.copy()

    # Ensure required columns exist
      if all(col in df.columns for col in ["CH_CLOSING_PRICE", "CH_TRADE_HIGH_PRICE", "CH_TRADE_LOW_PRICE"]):
        highest_high = df["CH_TRADE_HIGH_PRICE"].rolling(window=period, min_periods=1).max()
        lowest_low = df["CH_TRADE_LOW_PRICE"].rolling(window=period, min_periods=1).min()
        df["williams_%R"] = ((highest_high - df["CH_CLOSING_PRICE"]) / (highest_high - lowest_low)) * -100
      else:
        missing_cols = [col for col in ["CH_CLOSING_PRICE", "CH_TRADE_HIGH_PRICE", "CH_TRADE_LOW_PRICE"] if col not in df.columns]
        raise ValueError(f"Missing columns for Williams %R calculation: {missing_cols}")

      #print("After Williams %R calculation, columns:", df.columns)  # Debugging line
      return df


    @staticmethod
    def calculate_momentum_score(df: pd.DataFrame,
                                 rsi_weight: float = 0.3,
                                 macd_weight: float = 0.3,
                                 obv_weight: float = 0.1,
                                 stoch_weight: float = 0.1,
                                 williams_weight: float = 0.1) -> pd.DataFrame:
        """
        Calculates a momentum score by combining normalized signals from various indicators.
        Positive score suggests bullish momentum; negative indicates bearish.
        """
        df = df.copy()
        # Normalize RSI: transform to [-1, 1] where 0 is neutral (RSI=50)
        rsi_score = (df["rsi"] - 50) / 50

        # MACD difference normalized by its rolling standard deviation (z-score)
        macd_diff = df["macd"] - df["macd_signal"]
        macd_std = macd_diff.rolling(window=14, min_periods=1).std().replace(0, np.nan)
        macd_score = macd_diff / (macd_std + 1e-5)

        # OBV score: using the sign of the OBV difference
        obv_diff = df["obv"].diff().fillna(0)
        obv_score = np.sign(obv_diff)

        # Stochastic oscillator score: difference between %K and %D scaled to [-1,1]
        stoch_diff = df["stoch_%K"] - df["stoch_%D"]
        stoch_score = stoch_diff / 100.0

        # Williams %R score: normalize so that values close to 1 indicate bullish and -1 bearish
        williams_score = (df["williams_%R"] + 50) / 50.0 - 1

        # Combine scores using provided weights
        df["momentum_score"] = (rsi_weight * rsi_score +
                                macd_weight * macd_score +
                                obv_weight * obv_score +
                                stoch_weight * stoch_score +
                                williams_weight * williams_score)
        return df

    @staticmethod
    def add_trend_suggestions(df: pd.DataFrame, score_threshold: float = 0.5) -> pd.DataFrame:
        """
        Suggests bullish or bearish momentum based on the calculated momentum score.
        A momentum score above 'score_threshold' indicates bullish momentum,
        while below '-score_threshold' indicates bearish momentum.
        """
        df = df.copy()
        # Calculate momentum score
        df = DataPreprocessor.calculate_momentum_score(df)
        # Initialize trend as Neutral
        df["trend"] = 0
        df.loc[df["momentum_score"] > score_threshold, "trend"] = 1
        df.loc[df["momentum_score"] < -score_threshold, "trend"] = -1

    # Emoji version for visualization
        df["trend_emoji"] = "➖"  # Neutral
        df.loc[df["trend"] == 1, "trend_emoji"] = "📈"
        df.loc[df["trend"] == -1, "trend_emoji"] = "📉"
        return df

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all technical indicator methods to the DataFrame.
        """
        df = DataPreprocessor.clean_data(df)
        df = DataPreprocessor.add_daily_return(df)
        df = DataPreprocessor.add_moving_averages(df,windows=[7])
        df = DataPreprocessor.add_rsi(df)
        #print("Columns after adding RSI:", df.columns)
        df = DataPreprocessor.add_macd(df)
        df = DataPreprocessor.add_bollinger_bands(df)
        df = DataPreprocessor.add_adx(df)
        df = DataPreprocessor.add_obv(df)
        df = DataPreprocessor.add_stochastic(df)
        df = DataPreprocessor.add_williams_r(df)
        df = DataPreprocessor.calculate_momentum_score(df)

        df = DataPreprocessor.add_trend_suggestions(df)
        return df

    @staticmethod
    def validate_data(df: pd.DataFrame):
        """
        Validates the data for missing values and provides descriptive statistics.
        """
        print("\n🔎 Checking for missing values:")
        print(df.isnull().sum())

        print("\n📊 Data Description:")
        print(df.describe())

        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            print("⚠️ Warning: Missing values detected!")
        else:
            print("✅ No missing values detected.")
        return df

