%%writefile /content/drive/MyDrive/AI_stock_project/main.py
from datafetcher import DataFetcher
from datapreprocessor import DataPreprocessor
from backtester import Backtester


def main():
    try:
        # 1. Fetch Data
        fetcher = DataFetcher()
        raw_data = fetcher.get_historical_data("REDINGTON", days=30)

        if raw_data.empty:
            raise ValueError("Fetched data is empty. Check data source.")

        # 2. Preprocess Data
        processed_data = DataPreprocessor.add_all_indicators(raw_data)

        if processed_data.empty:
            raise ValueError("No data remaining after preprocessing. Check indicators.")

        # 3. Predict and Backtest
        today_prediction = DataPreprocessor.predict_today_trend(processed_data)
        latest_date = processed_data["date"].iloc[-1].strftime("%Y-%m-%d")
        print(f"Predicted Trend for {latest_date}: {today_prediction}")

        # Optional backtest
        backtester = Backtester(processed_data)
        backtest_results = backtester.simulate_trades()
        print("\nBacktest Results (Last 5 Days):")
        print(backtest_results[["date", "portfolio_value", "trend_emoji"]]
              .tail()
              .round({"portfolio_value": 3})  # Round to 3 decimals
              .to_string(index=False))

        #print("\nBacktest Summary:")
        #print(f"Final Portfolio Value:{backtest_results['portfolio_value'].iloc[-1]:.2f}")

    except Exception as e:
        print(f"🚨 Error: {str(e)}")


if __name__ == "__main__":
    main()