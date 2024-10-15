import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator, DayLocator
import plotly.graph_objects as go
import seaborn as sns
import time
import datetime

def get_stock_data(ticker):
    url = f'https://finance.yahoo.com/quote/{ticker}?p={ticker}'
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract the stock price
    price_section = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
    stock_price = float(price_section.text.replace(',', '')) if price_section else None

    # Extract other data points like P/E ratio
    pe_ratio_section = soup.find('td', {'data-test': 'PE_RATIO-value'})
    pe_ratio = pe_ratio_section.text if pe_ratio_section else 'N/A'

    # Return the extracted data
    return {'Stock': ticker, 'Price': stock_price, 'P/E Ratio': pe_ratio}

# print(get_stock_data('SUZLON.NS'))

def date_to_unix(date_string):
    # Convert the date string to a datetime object
    date_object = datetime.datetime.strptime(date_string, '%Y-%m-%d')
    
    # Convert the datetime object to a Unix timestamp
    unix_timestamp = int(time.mktime(date_object.timetuple()))
    
    return unix_timestamp

def get_stock_historical_data(ticker, start_date, end_date):
    # Recommended to validate date string
    period1 = date_to_unix(start_date)
    period2 = date_to_unix(end_date)

    url = f'https://finance.yahoo.com/quote/{ticker}/history?period1={period1}&period2={period2}'
    headers = {"User-Agent": "Mozilla/5.0"}

    # Extract history
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table containing historical data
    table = soup.find('table', {'class': 'yf-ewueuo'})

    # Get the table headers (And remove extra whitespace and html elements)
    # headers = [header.get_text(strip=True, recursive=) for header in table.find('thead').find_all('th')]
    headers = [header.find(string=True, recursive=False).strip().replace(',', '') for header in table.find('thead').find_all('th')]

    # Get the table rows (each row contains historical data for a specific date)
    rows = []
    for row in table.find('tbody').find_all('tr'):
        cols = [col.find(string=True, recursive=False).strip().replace(',', '') for col in row.find_all('td')]
        rows.append(cols)
    
    # Create a DataFrame with the scraped data
    df = pd.DataFrame(rows, columns=headers)
    
    return df

# Function to summarize x-axis based on the date range
def adjust_xaxis_format(ax, dates):
    date_span = (dates.max() - dates.min()).days  # Calculate the time span in days

    if date_span > 365:  # Data spans more than a year
        ax.xaxis.set_major_locator(YearLocator())  # Use years for labels
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))  # Format as 'Year'
    elif date_span > 30:  # Data spans between a month and a year
        ax.xaxis.set_major_locator(MonthLocator())  # Use months for labels
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))  # Format as 'Month Year'
    else:  # Data spans less than a month
        ax.xaxis.set_major_locator(DayLocator())  # Use days for labels
        ax.xaxis.set_major_formatter(DateFormatter('%d %b'))  # Format as 'Day Month'

# Function for basic analysis
def basic_analysis(csv_file):
    """
    Perform basic analysis like calculating moving averages and returns.
    :param stock_df: DataFrame containing stock data
    :return: None
    """
    stock_df = pd.read_csv("./"+csv_file)
    
    # Convert the 'Date' column to datetime format without a specific format
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')

    # Check if there are any NaT (Not a Time) values after conversion
    if stock_df['Date'].isnull().any():
        print("Some dates could not be converted. Please check the format.")

    # 1. Candlestick Chart (Recommended for stock data)
    # Use Case: This is ideal for visualizing the daily open, high, low, and close prices. It helps show price movements throughout each day.
    # How to Plot: Each candlestick represents one day. The "body" shows the open and close prices, while the "wicks" or "shadows" show the high and low prices.
    # Library: plotly or mplfinance in Python.
    fig = go.Figure(data=[go.Candlestick(x=stock_df['Date'],
                                     open=stock_df['Open'],
                                     high=stock_df['High'],
                                     low=stock_df['Low'],
                                     close=stock_df['Close'])])

    fig.update_layout(title='Stock Price Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    fig.show()

    # 2. Line Chart
    # Use Case: Use a line chart to display trends over time, such as how the adjusted close price changes.
    # How to Plot: Plot the dates on the x-axis and the adjusted close prices on the y-axis.
    # Library: matplotlib or plotly.

    # Plot the data
    fig1, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_df['Date'], stock_df['Adj Close'], label='Adj Close')

    # Adjust the x-axis format based on the data span
    adjust_xaxis_format(ax, stock_df['Date'])

    # Set other labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Adjusted Close Price')
    ax.set_title('Adjusted Close Price Over Time')
    ax.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.show()

    # 3. Volume Bar Chart
    # Use Case: Visualize the trading volume per day to show how much stock was traded on each date.
    # How to Plot: A bar chart with dates on the x-axis and volume on the y-axis.
    # Library: matplotlib or plotly.
    plt.figure(figsize=(10, 5))
    plt.bar(stock_df['Date'], stock_df['Volume'], color='lightblue')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.title('Trading Volume Over Time')
    plt.xticks(rotation=45)
    plt.show()

    # 4. Moving Average Line Chart
    # Use Case: Show the 50-day and 200-day moving averages to indicate long-term and short-term price trends.
    # How to Plot: Similar to a line chart, but add the moving averages as additional lines.
    # Library: matplotlib or plotly
    
    # stock_df['Adj Close'] = pd.to_numeric(stock_df['Adj Close'], errors='coerce')

    # Calculate moving averages (50-day and 200-day)
    stock_df['50_MA'] = stock_df['Adj Close'].rolling(window=50).mean()
    stock_df['200_MA'] = stock_df['Adj Close'].rolling(window=200).mean()

    # Calculate daily returns
    stock_df['Daily_Return'] = stock_df['Adj Close'].pct_change()

    print("\n=== Summary Statistics ===")
    print(stock_df.describe())

    # Plot Adjusted Close Price and Moving Averages
    plt.figure(figsize=(14, 7))
    plt.plot(stock_df['Adj Close'], label='Adjusted Close Price', color='blue')
    plt.plot(stock_df['50_MA'], label='50-Day Moving Average', color='green')
    plt.plot(stock_df['200_MA'], label='200-Day Moving Average', color='red')
    plt.title('Adjusted Close Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.show()

    # Plot Daily Returns
    plt.figure(figsize=(10, 5))
    sns.histplot(stock_df['Daily_Return'].dropna(), bins=50, color='purple')
    plt.title('Daily Returns Distribution')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.show()

# Function to analyze a specific stock
def analyze_stock(stock_symbol, start_date='2020-01-01', end_date='2024-10-15'):
    """
    Download data and perform basic analysis for the given stock.
    :param stock_symbol: Stock symbol (e.g., 'TCS.NS')
    :param start_date: Start date for analysis
    :param end_date: End date for analysis
    :return: None
    """
    # Step 1: Get stock data
    print(f"Fetching data for {stock_symbol} from Yahoo Finance...")
    stock_df = get_stock_historical_data(stock_symbol, start_date, end_date)

    print(stock_df)
    csv_filename = f"stock_data_{stock_symbol}.csv"
    stock_df.to_csv(csv_filename, index=True)

    # Step 2: Perform basic analysis
    basic_analysis(csv_filename)

# Example: Analyzing Tata Consultancy Services (TCS) stock
if __name__ == "__main__":
    # Example stock symbols for Indian companies
    stock_symbol = 'SUZLON.NS'  # Tata Consultancy Services (TCS) on NSE
    analyze_stock(stock_symbol, start_date='2020-01-01', end_date='2024-10-15')

# symbol = 'SUZLON.NS'
# df = get_stock_history(symbol, '2020-10-01', '2024-10-14')

# # print(df)
# # Export to CSV
# csv_filename = f"historical_data_{symbol}.csv"
# df.to_csv(csv_filename, index=False)
# # print(f"Data exported to {csv_filename}")
# basic_analysis(csv_filename)





