import pandas as pd
import requests
from bs4 import BeautifulSoup

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

def get_stock_history(ticker):
    url = f'https://finance.yahoo.com/quote/{ticker}/history/'
    headers = {"User-Agent": "Mozilla/5.0"}

    # Extract history
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table containing historical data
    table = soup.find('table', {'class': 'yf-ewueuo'})
    
    # Get the table headers
    headers = [header.text for header in table.find('thead').find_all('th')]
    
    # Get the table rows (each row contains historical data for a specific date)
    rows = []
    for row in table.find('tbody').find_all('tr'):
        cols = [col.text for col in row.find_all('td')]
        rows.append(cols)
    
    # Create a DataFrame with the scraped data
    df = pd.DataFrame(rows, columns=headers)
    
    return df

symbol = 'SUZLON.NS'
df = get_stock_history(symbol)

csv_filename = f"historical_data_{symbol}.csv"
df.to_csv(csv_filename, index=False)
print(f"Data exported to {csv_filename}")



