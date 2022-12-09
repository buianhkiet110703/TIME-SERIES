# TIME-SERIES

# I. Giới Thiệu Dataset
Đây là tập dữ liệu về Bitcoin bao gồm các cột:

     - Open: Price from the first transaction of a trading day
     
     - High: Maximum price in a trading day
     
     - Low: Minimum price in a trading day
     
     - Close: Price from the last transaction of a trading day
     
     - Adj Close: Closing price adjusted to reflect the value after accounting for any corporate actions
     
     - Volume: Number of units traded in a day
     
 Dataset này cũng có sẵn trong python dùng 'import yfinance' để lấy dataset.
 
 Với dataset này thì sẽ dùng các mô hình time series để dự đoán chúng , trong bài này sẽ dùng 3 mô hình đó : ARIMA
 
 # II. DATA CLEANING AND VISUALIZE
 
   Đầu tiên chúng ta kiểm tra xem có cột nào dữ liệu bị NAN hay không.
   ```php
   df.isnull().sum()
   ```
   output:
   
   ![](https://scontent.fsgn2-5.fna.fbcdn.net/v/t1.15752-9/313457409_1767206376981273_6702053681674880099_n.png?_nc_cat=106&ccb=1-7&_nc_sid=ae9488&_nc_ohc=BBfmdA5T1qsAX_KaUiE&_nc_ht=scontent.fsgn2-5.fna&oh=03_AdR2QStrm2fwrLbyCUrSSVJE9kVuitkUV_8zvqAfSlbblg&oe=63947F3C)
   
Dữ liệu rất là sạc sẽ không có dữ liệu bị NAN.
   
   Vẽ biểu đồ scatter để xem sự tương quan của các biến với nhau.
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
 
![](https://scontent.fsgn2-8.fna.fbcdn.net/v/t1.15752-9/313786780_512466404137844_6942073010345852627_n.png?_nc_cat=102&ccb=1-7&_nc_sid=ae9488&_nc_ohc=mZHbJP7iKHUAX9pf8qY&_nc_ht=scontent.fsgn2-8.fna&oh=03_AdTdec_7SovSfYSyuqVoB2tiP-U5BHka67epy85epH8Rvg&oe=6395CCC5)
   

Vẽ biểu đồ xem thử dữ liệu thô ban đầu của dataset.
```php
plt.figure(figsize=(12,6))
plt.grid(True)
plt.plot(x)
plt.xlabel('Date')
plt.ylabel('Adj Close')
plt.title('biểu đồ thể hiện Adj Close theo các năm')
```
output:

![](https://scontent.fsgn2-7.fna.fbcdn.net/v/t1.15752-9/308809403_481410593973806_2064451325614451912_n.png?_nc_cat=109&ccb=1-7&_nc_sid=ae9488&_nc_ohc=TqE4kkvR_mIAX-a6LXI&_nc_ht=scontent.fsgn2-7.fna&oh=03_AdTKm5X4XYQ70Cenci8U6FthcN5N3fKJ7K0oXXTE60bZ1w&oe=63947874)

Tiếp theo Tôi sẽ chia chia dữ liệu thành tập train và tập test để xây dựng mô hình ARIMA.

```php
to_row= int(len(x)*0.8)
training_data= list(x[0: to_row])
testing_data= list(x[to_row:])
```
Vẽ biểu đồ xem phần train và phần test sau khi tách ra.

```php
plt.figure(figsize=(12,6))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('ADJ close')
plt.plot(x[0: to_row],'green', label= 'training_data')
plt.plot(x[to_row:],'blue', label= 'testing_data')
plt.legend()
```
![image](https://user-images.githubusercontent.com/110837675/206652189-48d374e1-d0a0-40eb-9fbc-5c029229cf1f.png)

Vẽ biểu đồ AFC và PACF xem sự tương quan của Lag_time

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

# Build Model

Dùng auto arima để tìm ra được AIC tốt nhất cho mô hình.

```php
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')
stepwise_fit= auto_arima(training_data,trace= True, suppress_warnings= True)

stepwise_fit.summary()
```
![image](https://user-images.githubusercontent.com/110837675/206654350-a1ab4da8-e5fc-4dc0-8103-86cb8eadd360.png)

Sau khi có được tham số tốt nhất cho mô hình thì bắt đầu xây dựng mô hình.

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
Đánh giá mô hình bằng mean_squared_error.

```php
mse=mean_squared_error(testing_data, predictions)
rmse = sqrt(mse)
print('MSE: %.2f' % mse)
print('RMSE: %.2f' % rmse)
```
![image](https://user-images.githubusercontent.com/110837675/206654812-9277cef3-5747-48a1-8f73-1419427acb45.png)

Vẽ biểu đồ của giá thực tế so với giá dự đoán.

```php
plt.figure(figsize=(12,6))
plt.grid(True)
data_range= df[to_row:].index
plt.plot(data_range, predictions, color='orange' , label='BTC predict')
plt.plot(data_range, testing_data, color='blue', label='BTC acctual')
plt.legend()
```
![image](https://user-images.githubusercontent.com/110837675/206654995-f2302e54-4f72-4e5d-8c98-97203bf17762.png)

Đường giá dự đoán và đường giá thực tế cũng khá là gần nhau có thể thấy mô hình này có thể hoạt động tốt.





   

