import pandas as pd

data = pd.read_csv('../dataset/rossmann.csv', skipinitialspace=True, low_memory=False)

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data['Year'] = data.index.year
data['Month'] = data.index.month

store1_data = data[data.Store == 1]

import seaborn as sb
%matplotlib inline

sb.factorplot(
    x='SchoolHoliday',
    y='Sales',
    data=store1_data,
    kind='box'
)
# Check: See if there is a difference affecting sales on promotion days.

sb.factorplot(
    x='Promo',
    y='Sales',
    data=store1_data,
    kind='box'
)

# Filter to days store 1 was open
store1_open_data = store1_data[store1_data.Open==1]
store1_open_data[['Sales']].plot()

store1_open_data[['Customers']].plot()

data['Sales'].resample('D').mean().autocorr(lag=1)

data[['Sales']].resample('M').apply(['median', 'mean']).head()

data[['Sales']].resample('D').mean().rolling(window=3, center=True).mean().head()

data[['Sales']].resample('D').mean().rolling(window=3, center=True).mean().plot()

data['Sales'].diff(periods=1).head()

# computes the average sales, from the first date _until_ the date specified.
data[['Sales']].resample('D').mean().expanding().mean().head()

# Plot the distribution of sales by month and compare the effect of promotions
store1_open_data[['Sales']].plot()
sb.factorplot(
    col='Open',
    hue='Promo',
    x='Month',
    y='Sales',
    data=store1_data,
    kind='box'
)

# Are sales more correlated with the prior date, a similar date last year,
# or a similar date last month?

data['Sales'].resample('D').mean().autocorr(lag=1)

data['Sales'].resample('D').mean().autocorr(lag=30)

data['Sales'].resample('D').mean().autocorr(lag=365)

# Plot the 15 day rolling mean of customers in the stores
data[['Customers']].resample('D').mean().rolling(window=15, center=True).mean().plot()
data[['Customers']].resample('D').mean().rolling(window=15, center=True).mean().[:10]

# Identify the date with largest drop in sales from the same date in the previous month
data[['Sales']].resample('D').mean().diff(30).max()
data[['Sales']].resample('D').mean().diff(30).sort_values(by='Sales')
