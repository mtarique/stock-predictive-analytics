import yfinance as yf
import matplotlib.pyplot as plt

data = yf.download('AAPL', start='2020-01-01', end='2024-02-10')
data['Close'].plot(title="AAPL Closing Prices")
plt.show()

'''
data['Daily Return'] = data['Close'].pct_change()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Prepare data
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

plt.plot(y_test, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.legend()
plt.show()
'''
