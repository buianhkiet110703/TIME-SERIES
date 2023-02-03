## Time_series with Arima and  Long short term memory (LSTM)

# Introduce Dataset 

Here is the Bitcoin dataset including the columns: 

     - Open: Price from the first transaction of a trading day
     
     - High: Maximum price in a trading day
     
     - Low: Minimum price in a trading day
     
     - Close: Price from the last transaction of a trading day
     
     - Adj Close: Closing price adjusted to reflect the value after accounting for any corporate actions
     
     - Volume: Number of units traded in a day
     
 This dataset is also available in python using 'import yfinance' to get the dataset. In this dataset, I will Predict with 'Adj Close' Columns.
 
# A. MODEL ARIMA
 
 # I. DATA CLEANING AND VISUALIZE
 
First we check to see if any columns of data are NAN.
   ```php
   df.isnull().sum()
   ```
   output:
   
![image](https://user-images.githubusercontent.com/110837675/215991013-9c531f77-abd3-4329-a786-8b6a5cf3129a.png)

   
Very charging data will not have NAN data.
   
Draw a scatter plot to see the correlation of the variables with each other.
```php
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6))= plt.subplots(nrows=2, ncols=3,figsize=(20,10))
ax1.scatter(df['Adj Close'], df['Open'],c='blue')
ax1.set(title='biểu đồ tương qua giữa Adj Close và Open', xlabel='Adj Close', ylabel='open');
ax2.scatter(df['Adj Close'], df['High'],c='green')
ax2.set(title='biểu đồ tương qua giữa Adj Close và High', xlabel='Adj Close', ylabel='High');
ax3.scatter(df['Adj Close'], df['Low'],c='gray')
ax3.set(title='biểu đồ tương qua giữa Adj Close và Low', xlabel='Adj Close', ylabel='Low');
ax4.scatter(df['Adj Close'], df['Close'],c='brown')
ax4.set(title='biểu đồ tương qua giữa Adj Close và Close', xlabel='Adj Close', ylabel='Close');
ax5.scatter(df['Adj Close'], df['Volume'],c='black')
ax5.set(title='biểu đồ tương qua giữa Adj Close và Volume', xlabel='Adj Close', ylabel='Volume');
```
 Output:
 
![image](https://user-images.githubusercontent.com/110837675/215991378-da945b73-e949-430d-9ee5-16e696e5d7a8.png)
   
   
Plot a preview of the dataset's original raw data.
```php
plt.figure(figsize=(12,6))
plt.grid(True)
plt.plot(x)
plt.xlabel('Date')
plt.ylabel('Adj Close')
plt.title('chart showing Adj Close by years')
```
output:

![image](https://user-images.githubusercontent.com/110837675/215991965-94af5130-5bd5-4642-b5a5-5328d1871d54.png)

Next, I will divide the data into a train set and a test set to build the ARIMA model.

```php
to_row= int(len(x)*0.8)
training_data= list(x[0: to_row])
testing_data= list(x[to_row:])
```

Draw a graph to see the train part and the test part after separating.

```php
plt.figure(figsize=(12,6))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('ADJ close')
plt.title('Train/Test Split')
plt.plot(x[0: to_row],'green', label= 'training_data')
plt.plot(x[to_row:],'blue', label= 'testing_data')
plt.legend()
```
![image](https://user-images.githubusercontent.com/110837675/215992425-43620b3b-fd1e-4df5-996f-0bc17f674438.png)

Draw an AFC and PACF graph to see the correlation of Lag_time

```php
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(x.values) #Hàm vẽ ACF
plt.grid(linestyle='--')
plt.ylabel('Correlation',fontsize=12)
plt.xlabel('Lag-time',fontsize=12)
plt.show()
```
![image](https://user-images.githubusercontent.com/110837675/206653681-11bf5ba1-36ba-463b-9e02-b571580e031f.png)

```php
plot_pacf(x.values) #Hàm vẽ PACF
plt.grid(linestyle='--')
plt.ylabel('Correlation',fontsize=12)
plt.xlabel('Lag-time',fontsize=12)
plt.show()
```
![image](https://user-images.githubusercontent.com/110837675/206653921-bf4ddcbd-ad6c-425a-bf5f-45d4a1140263.png)

# II. Build Model

Use auto arima to find the best AIC for the model.

```php
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')
stepwise_fit= auto_arima(training_data,trace= True, suppress_warnings= True)

stepwise_fit.summary()
```
![image](https://user-images.githubusercontent.com/110837675/206654350-a1ab4da8-e5fc-4dc0-8103-86cb8eadd360.png)

After getting the best parameter for the model, start building the model.

```php 
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
history = [i for i in training_data]
predictions = list()
#Dự báo 1 bước thời gian
for t in range(len(testing_data)):
	model = ARIMA(history, order=(5,2,0)) #cài đặt p(AR), d(I), q(MA)
	model_fit = model.fit()
	output = model_fit.forecast()# Dự báo cho toàn bộ chuỗi thời gian
	yhat = output[0] # Giá trị dự báo t+1
	predictions.append(yhat)
	obs = testing_data[t]
	history.append(obs)
	print('Giá trị dự báo=%f, Giá trị thực tế=%f' % (yhat, obs)) # Hiển thị giá trị dự báo của từng bước thời gian 
```

Evaluate the model using mean_squared_error.

```php
mse=mean_squared_error(testing_data, predictions)
rmse = sqrt(mse)
print('MSE: %.2f' % mse)
print('RMSE: %.2f' % rmse)
```
![image](https://user-images.githubusercontent.com/110837675/206654812-9277cef3-5747-48a1-8f73-1419427acb45.png)

Plot a chart of the actual price versus the predicted price.

```php
plt.figure(figsize=(12,6))
plt.grid(True)
data_range= df[to_row:].index
plt.plot(data_range, predictions, color='orange' , label='BTC predict')
plt.plot(data_range, testing_data, color='blue', label='BTC acctual')
plt.legend()
```
![image](https://user-images.githubusercontent.com/110837675/206654995-f2302e54-4f72-4e5d-8c98-97203bf17762.png)


The predicted price line and the actual price line are also quite close to each other, which shows that this model can work well.

# B. Long short term memory (LSTM)

# I.Data preprocessing

I will split the 'Adj close' column data from the original dataset into a new one. Then, I will Scaler data.
```php
sequence = df['Adj Close'].to_frame()
from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1))
scaled= sc.fit_transform(sequence)

sequence= scaled
```
I will split Training and Testing data.
```php
timestep = 10
x_train, y_train = [], []
for i in range(timestep, len(train)):
    x_train.append(train[i - timestep:i])
    y_train.append(train[i])

x_test, y_test = [], []
for i in range(timestep, len(test)):
    x_test.append(test[i - timestep:i])
    y_test.append(test[i])
```
This code is used to generate input data for the prediction model. First, we define a variable timestep = 10, that is, we will use the previous 10 values of the value in the train set to predict the next value.

Then we use two for loops to generate the input data. The first loop uses train set to generate x_train and y_train data, for each i value in train set from timestep to end of set, we will create an element x_train which is a list of 10 previous values of train set and an element y_train is the value of train set at position i.

The second loop creates the same as the first, but uses the test set instead of the train set.

I will convert List into array.

```php
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
```

This code changes the shape of the set x_train and x_test from 2-dimensional to 3-dimensional. x_train.shape[0] returns the number of rows, x_train.shape[1] returns the number of columns, and 1 is the number of dimensions (number of features or number of channels). The new shape of x_train and x_test is the number of rows x number of columns x 1, which helps the LSTM model to recognize that the data being processed is temporal data with one dimension.
```php
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
```

# III. Build Model

```php
model =Sequential([LSTM(units=100, input_shape=(timestep, 1)),Dense(1)])
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)
```
# IV. Evaluate Model

Return data to original value.

```php
y_train= sc.inverse_transform(y_train)
y_pred_train = model.predict(x_train)
y_pred_train= sc.inverse_transform(y_pred_train)
y_test= sc.inverse_transform(y_test)
y_pred_test = model.predict(x_test)
y_pred_test=sc.inverse_transform(y_pred_test)
```

Plot Actual Value compare with Predictions.
```php
plt.figure(figsize=(12,6))
plt.plot(df.index[:len(y_train)], y_train, label='Real Train')
plt.plot(df.index[:len(y_pred_train)], y_pred_train, label='Prediction (Train)')
start_index = len(y_train)
end_index = start_index + len(y_test)
plt.plot(df.index[start_index:end_index], y_test, label='Real Test')
plt.plot(df.index[start_index:end_index], y_pred_test, label='Prediction (Test)')
plt.xlabel("Index")
plt.ylabel("Price")
plt.title('Actual Value compare with Predictions')
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/110837675/216515294-06fd7c12-6fc5-4932-8d51-e2c479800f68.png)

Mean_squared_error of model.
![image](https://user-images.githubusercontent.com/110837675/216515357-985a91ec-3c00-4f3f-9579-b6b0fe7b8b31.png)

# V. Predict next 6 months future.
```php
# Tính toán dữ liệu đầu vào cho 6 tháng tiếp theo
from datetime import timedelta
last_6month = y_test[-180:]
input_data = []
for i in range(len(last_6month) - timestep):
    input_data.append(last_6month[i:i + timestep])
input_data = np.array(input_data).reshape(-1, timestep, 1)

# Dự đoán giá trị 6 tháng tiếp theo
y_pred_6month = model.predict(input_data)
y_pred_6month= sc.inverse_transform(y_pred_6month)
# Tạo DataFrame cho dữ liệu dự đoán
df_pred_6month = pd.DataFrame({'Price': y_pred_6month.flatten()})

# Tạo cột "Date" tương ứng với dữ liệu dự đoán
last_date = df.index[-1]
df_pred_6month['Date'] = [last_date + timedelta(days=i) for i in range(1, len(df_pred_6month) + 1)]
```
Plot Predict next 6 months future:

```php
# Vẽ hình
plt.figure(figsize=(12,6))
plt.plot(df_pred_6month['Date'], df_pred_6month['Price'], label='Prediction (6 Months)')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title('Predict next 6 months future')
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/110837675/216515479-ca1e4382-3172-47db-ba93-b89d11a526ca.png)

```php
plt.figure(figsize=(15,10))

plt.subplot(3,1,1)
plt.plot(df.index[:len(y_train)], y_train, label='Real Train')
plt.plot(df.index[:len(y_pred_train)], y_pred_train, label='Prediction (Train)')
plt.xlabel("Index")
plt.ylabel("Price")
plt.legend()

plt.subplot(3,1,2)
start_index = len(y_train)
end_index = start_index + len(y_test)
plt.plot(df.index[start_index:end_index], y_test, label='Real Test')
plt.plot(df.index[start_index:end_index], y_pred_test, label='Prediction (Test)')
plt.xlabel("Index")
plt.ylabel("Price")
plt.legend()

plt.subplot(3,1,3)
plt.plot(df_pred_6month['Date'], df_pred_6month['Price'], label='Prediction (6 Months)')
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

plt.show()
```
![image](https://user-images.githubusercontent.com/110837675/216515630-f3051c5e-7071-4a81-84f0-052d6f326a3c.png)

   

