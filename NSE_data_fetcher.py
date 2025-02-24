
#%%writefile /content/drive/MyDrive/AI_stock_project/datafetcher.py
from nsepython import *
import pandas as pd
from datetime import datetime, timedelta
from google.colab import files

class DataFetcher:
    def __init__(self):
        pass

    def get_historical_data(self, symbol: str, days: int = 30, save_csv: bool = False):

        end_date = datetime.now().strftime('%d-%m-%Y')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%d-%m-%Y')

        data = equity_history(symbol,"EQ", start_date, end_date)
        df = pd.DataFrame(data)


        if save_csv:
          filename = f"{symbol}_historical_{end_date}.csv"
          df.to_csv(filename, index=False)
          print(f"Data saved as {filename}")


          files.download(filename)

        return df

if __name__ == "__main__":
    fetcher = DataFetcher()
    df = fetcher.get_historical_data("REDINGTON", days=30, save_csv=True)
    print(df.head())
