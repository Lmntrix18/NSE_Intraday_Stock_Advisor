%%writefile /content/drive/MyDrive/AI_stock_project/main.py
from datafetcher import DataFetcher
from datapreprocessor import DataPreprocessor
import pandas as pd

def fetch_stock_symbols_from_file(file_path: str) -> list:
    """
    Reads stock symbols from a text file.
    """
    try:
        with open(file_path, "r") as file:
            stock_symbols = [line.strip() for line in file if line.strip()]
        return stock_symbols
    except Exception as e:
        print(f"🚨 Error reading file {file_path}: {str(e)}")
        return []

def analyze_stocks(stock_symbols: list, days: int = 30):
    """
    Analyzes multiple stocks and categorizes them into bullish, bearish, and neutral trends.
    """
    results = []

    for symbol in stock_symbols:
        try:
            # 1. Fetch Data
            fetcher = DataFetcher()
            raw_data = fetcher.get_historical_data(symbol, days=days)

            if raw_data.empty:
                print(f"⚠️ No data found for {symbol}. Skipping.")
                continue

            # 2. Preprocess Data
            processed_data = DataPreprocessor.add_all_indicators(raw_data)

            if processed_data.empty:
                print(f"⚠️ No data remaining after preprocessing for {symbol}. Skipping.")
                continue

            # 3. Get Latest Momentum Score and Trend
            latest_score = processed_data["momentum_score"].iloc[-1]
            latest_date = processed_data["date"].iloc[-1].strftime("%Y-%m-%d")
            trend = DataPreprocessor.predict_today_trend(processed_data)

            results.append({
                "symbol": symbol,
                "latest_date": latest_date,
                "momentum_score": latest_score,
                "trend": trend
            })

        except Exception as e:
            print(f"🚨 Error analyzing {symbol}: {str(e)}")

    # Convert results to DataFrame
    return pd.DataFrame(results)

def main():
    # Fetch stock symbols from a text file
    file_path = "/content/drive/MyDrive/AI_stock_project/stock_symbols.txt"  # Path to your text file
    stock_symbols = fetch_stock_symbols_from_file(file_path)

    if not stock_symbols:
        print("🚨 No stock symbols found. Exiting.")
        return

    print(f"Found {len(stock_symbols)} stocks in '{file_path}'.")

    # Analyze all stocks
    print("Analyzing stocks...")
    results_df = analyze_stocks(stock_symbols)

    # Categorize stocks into bullish, bearish, and neutral
    bullish_stocks = results_df[results_df["trend"] == "Bullish 📈"]
    bearish_stocks = results_df[results_df["trend"] == "Bearish 📉"]
    neutral_stocks = results_df[results_df["trend"] == "Neutral ➖"]

    # Display results
    print("\nBullish Stocks:")
    print(bullish_stocks[["symbol", "latest_date", "momentum_score", "trend"]]
          .to_string(index=False))

    print("\nBearish Stocks:")
    print(bearish_stocks[["symbol", "latest_date", "momentum_score", "trend"]]
          .to_string(index=False))

    print("\nNeutral Stocks:")
    print(neutral_stocks[["symbol", "latest_date", "momentum_score", "trend"]]
          .to_string(index=False))

if __name__ == "__main__":
    main()
