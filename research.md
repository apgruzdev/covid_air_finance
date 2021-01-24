```python
import os
import datetime
```

```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from scipy.signal import find_peaks

import plotly.graph_objects as go
import plotly.express as px

import requests
```

```python
path_2_data = r'./data'
```

```python
BASE_YAHOO_DATETIME = datetime.date(year=1970, month=1, day=1)
HISTORY_START_DATETIME = datetime.date(year=2015, month=1, day=1)
```

```python
YAHOO_TICKERS = \
    {'boeing': 'BA',
     'airbus': 'AIR.PA',
     'lockheed_martin': 'LMT',
     'northrop_grumman': 'NOC',
     'raytheon_technologies': 'RTX',
     'safran': 'SAF.PA',
     'rolls-royce': 'RR.L',
     'leonardo': 'LDO.MI',
     'bae_systems': 'BA.L',
     'delta_air_lines': 'DAL',
     'american_airlines_group': 'AAL',
     'lufthansa': 'LHA.DE',
     'united_airlines': 'UAL',
     'air_france': 'AF.PA',
     'iag': 'IAG.L',
     'southwest': 'LUV',
     'china_southern_airlines': '1055.HK',
     'all_nippon_airways': '9202.T',
     'china_eastern_airlines': '0670.HK'}
```

```python
YAHOO_URL = 'https://query1.finance.yahoo.com/v7/finance/download/{TICKER}?period1={END}&period2={START}&interval=1d&events=history'
```

```python
TICKER_COLUMN_NAME = 'ticker'
PRICE_COLUMN_NAME = 'Adj Close'
DATE_COLUMN_NAME = 'Date'
USE_COLUMNS = [DATE_COLUMN_NAME, PRICE_COLUMN_NAME, TICKER_COLUMN_NAME]

TRAND_COLUMN_NAME = 'trand'
SMOOTHED_COLUMN_NAME = 'smoothed'
PEAKS_COLUMN_NAME = 'peaks'
RISK_COLUMN_NAME = 'risk'
```

```python
LOCKDOWN_DATE = [datetime.datetime(year=2020, month=1, day=1), 
                 datetime.datetime(year=2020, month=3, day=23)]
```

```python
SMOOTH_WINDOW_SIZE = 30
SMOOTH_QUANTILE = 0.2

MINIMUM_SCALED_PROMINANCE = 0.05

NO_RISK_COEFF = 2.
```

```python
MONEY_AMOUNT = 10000
```

```python
# columns for final table
SHARE_COLUMN_NAME = 'share'
MONEY_SHARE = 'money_share'
EXPECTED_PROFIT = 'expected_money_profit'
EXPECTED_RISK_PROFIT = 'expected_risk_money_profit'
```

get data

```python
start_url = int((datetime.datetime.now().date() - BASE_YAHOO_DATETIME).total_seconds())
end_url = int((HISTORY_START_DATETIME - BASE_YAHOO_DATETIME).total_seconds())
```

```python
for company_name, ticker in YAHOO_TICKERS.items():
    request_url = YAHOO_URL.format(TICKER=ticker, START=start_url, END=end_url)
    response = requests.get(request_url, allow_redirects=True)
    
    with open(os.path.join(path_2_data, f'{company_name}.csv'), 'wb') as ticker_data_file:
        ticker_data_file.write(response.content)
```

plot

```python
dfs_tickers = []
for ticker_file_name in os.listdir(path_2_data):
    if ticker_file_name[0] == '.':
        continue
    
    df_ticker = pd.read_csv(os.path.join(path_2_data, ticker_file_name))
    df_ticker[TICKER_COLUMN_NAME] = ticker_file_name[:-4]
    dfs_tickers.append(df_ticker)
```

```python
tickers_data = pd.concat(dfs_tickers)
tickers_data.reset_index(inplace=True)
tickers_data = tickers_data[USE_COLUMNS]
tickers_data.loc[:, DATE_COLUMN_NAME] = pd.to_datetime(tickers_data[DATE_COLUMN_NAME])
```

```python
fig = px.line(tickers_data, x=DATE_COLUMN_NAME, y=PRICE_COLUMN_NAME, color=TICKER_COLUMN_NAME)
fig.write_html("avia_tickers_price.html")
```

sacling

```python
scaler = MinMaxScaler()
dfs_tickers_scaled = []
for ticker_data in tickers_data.groupby(TICKER_COLUMN_NAME):
    df_ticker = ticker_data[1].copy()
    df_ticker.loc[:, PRICE_COLUMN_NAME] = scaler.fit_transform(ticker_data[1][[PRICE_COLUMN_NAME]])
    dfs_tickers_scaled.append(df_ticker)
```

```python
tickers_data_scaled = pd.concat(dfs_tickers_scaled)
```

```python
fig = px.line(tickers_data_scaled, x=DATE_COLUMN_NAME, y=PRICE_COLUMN_NAME, color=TICKER_COLUMN_NAME)
fig.write_html("avia_tickers_scaled.html")
```

linear trand & risks

```python
def date_2_days(date_series: pd.Series) -> pd.Series:
    return (date_series - date_series.min())  / np.timedelta64(1, 'D')
```

```python
dfs_tickers_w_trand = []

for ticker_data in tickers_data_scaled.groupby(TICKER_COLUMN_NAME): 
    # add linear regression
    regressor = linear_model.LinearRegression()
    df_ticker = ticker_data[1].fillna(method='ffill').copy()
    fit_timeline_mask = df_ticker[DATE_COLUMN_NAME] < LOCKDOWN_DATE[0]
    
    df_ticker_fit = df_ticker.loc[fit_timeline_mask, :]
    days_timeline = date_2_days(df_ticker_fit[DATE_COLUMN_NAME])
    regressor.fit(pd.DataFrame(days_timeline), df_ticker_fit[PRICE_COLUMN_NAME])
    
    if regressor.coef_[0] < 0:
        print(f"negative ticker {ticker_data[0]}")
        continue
    
    df_ticker[TRAND_COLUMN_NAME] = regressor.predict(pd.DataFrame(date_2_days(df_ticker[DATE_COLUMN_NAME])))
    
    # add down peaks
    smoothed = df_ticker[PRICE_COLUMN_NAME].rolling(window=SMOOTH_WINDOW_SIZE, min_periods=None, center=True).quantile(SMOOTH_QUANTILE)
    df_ticker[SMOOTHED_COLUMN_NAME] = smoothed
    
    df_ticker[RISK_COLUMN_NAME] = np.nan
    peaks_indexes = find_peaks(-smoothed, prominence=[MINIMUM_SCALED_PROMINANCE, 1])[0]
    peaks = np.array([np.nan] * df_ticker.shape[0])
    peaks[peaks_indexes] = smoothed.iloc[peaks_indexes]

    df_ticker[PEAKS_COLUMN_NAME] = peaks
    filter_mask = df_ticker[PEAKS_COLUMN_NAME] > df_ticker[TRAND_COLUMN_NAME]
    df_ticker.loc[filter_mask, PEAKS_COLUMN_NAME] = np.nan
    
    # add risk linear regression
    regressor_risk = linear_model.LinearRegression()
    df_ticker_fit = df_ticker.loc[fit_timeline_mask, :]
    
    fit_risk_mask = df_ticker_fit[PEAKS_COLUMN_NAME].notnull()
    df_ticker_fit = df_ticker_fit.loc[fit_risk_mask, :]
    if df_ticker_fit.empty:
        print(f"no risk for {ticker_data[0]}")
        df_ticker[RISK_COLUMN_NAME] = df_ticker[TRAND_COLUMN_NAME] / NO_RISK_COEFF
    else:
        days_timeline = days_timeline.loc[fit_risk_mask]
        regressor_risk.fit(pd.DataFrame(days_timeline), df_ticker_fit[PEAKS_COLUMN_NAME])
        df_ticker[RISK_COLUMN_NAME] = regressor_risk.predict(pd.DataFrame(date_2_days(df_ticker[DATE_COLUMN_NAME])))
    
    dfs_tickers_w_trand.append(df_ticker)
```

```python
tickers_data_w_trand = pd.concat(dfs_tickers_w_trand)
```

plot 

```python
fig_main = px.line(tickers_data_w_trand, x=DATE_COLUMN_NAME, 
              y=[PRICE_COLUMN_NAME, TRAND_COLUMN_NAME],
              color=TICKER_COLUMN_NAME)

fig_risk = px.line(tickers_data_w_trand, x=DATE_COLUMN_NAME,
                   y=[SMOOTHED_COLUMN_NAME, RISK_COLUMN_NAME],  # RISK_COLUMN_NAME
                   color=TICKER_COLUMN_NAME, color_discrete_sequence=px.colors.qualitative.Pastel1)
for data_chunk in fig_risk.data:
    fig_main.add_trace(data_chunk)
    
fig_peaks = px.scatter(tickers_data_w_trand, x=DATE_COLUMN_NAME,
                       y=[PEAKS_COLUMN_NAME],  # RISK_COLUMN_NAME
                       color=TICKER_COLUMN_NAME, color_discrete_sequence=px.colors.qualitative.Dark2)
for data_chunk in fig_peaks.data:
    fig_main.add_trace(data_chunk)

fig_main.write_html("avia_tickers_w_trand.html")
```

```python
expected_profit = {}
for ticker_data in tickers_data_w_trand.groupby(TICKER_COLUMN_NAME): 
    ticker_name = ticker_data[0]
    
    df_ticker = ticker_data[1]
    expectstion_df = df_ticker[df_ticker[DATE_COLUMN_NAME] > LOCKDOWN_DATE[1]]
    
    expected_prices = expectstion_df[[PRICE_COLUMN_NAME, TRAND_COLUMN_NAME, RISK_COLUMN_NAME]].median()
    
    expected_profit[ticker_name] = {TRAND_COLUMN_NAME: (expected_prices[TRAND_COLUMN_NAME] - 
                                                        expected_prices[PRICE_COLUMN_NAME]) / expected_prices[PRICE_COLUMN_NAME], 
                                    RISK_COLUMN_NAME: (expected_prices[RISK_COLUMN_NAME] - 
                                                       expected_prices[PRICE_COLUMN_NAME]) / expected_prices[PRICE_COLUMN_NAME]}
```

compute shares for diversified profit

```python
expected_profit_df = pd.DataFrame(expected_profit).T
expected_profit_df = expected_profit_df.sort_values(by=TRAND_COLUMN_NAME, ascending=False)

sum_shares = expected_profit_df[TRAND_COLUMN_NAME].sum()
expected_profit_df[SHARE_COLUMN_NAME] = (expected_profit_df[TRAND_COLUMN_NAME] / sum_shares).round(2)
expected_profit_df[MONEY_SHARE] = (expected_profit_df[SHARE_COLUMN_NAME] * MONEY_AMOUNT).astype(int)

expected_profit_df[EXPECTED_PROFIT] = (expected_profit_df[TRAND_COLUMN_NAME] * expected_profit_df[MONEY_SHARE]).astype(int)
expected_profit_df[EXPECTED_RISK_PROFIT] = (expected_profit_df[RISK_COLUMN_NAME] * expected_profit_df[MONEY_SHARE]).astype(int)
```

```python
expected_profit_df.to_csv(f"shares_for_{MONEY_AMOUNT}.csv", sep=';')
```

```python
expected_profit_df
```

```python
f"expected profit: {expected_profit_df[EXPECTED_PROFIT].sum()}"
```

```python
f"expected risk profit: {expected_profit_df[EXPECTED_RISK_PROFIT].sum()}"
```

```python

```
