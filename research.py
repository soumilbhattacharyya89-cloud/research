import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

np.random.seed(1)
days = 500
aqi = 70 + 10*np.sin(np.linspace(0,20,days)) + np.random.normal(0,2,days)

df = pd.DataFrame({"AQI": aqi})

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

X, y = [], []
seq_len = 10
for i in range(len(scaled)-seq_len):
    X.append(scaled[i:i+seq_len])
    y.append(scaled[i+seq_len])

X, y = np.array(X), np.array(y)

split = int(0.8*len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(seq_len,1)),
    LSTM(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

pred = model.predict(X_test)
pred = scaler.inverse_transform(pred)
real = scaler.inverse_transform(y_test.reshape(-1,1))

plt.plot(real, label="Actual AQI")
plt.plot(pred, label="Predicted AQI")
plt.legend()
plt.show()
