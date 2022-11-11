# TIME-SERIES
# I. Giới Thiệu Dataset
Đây là tập dữ liệu về Bitcoin bao gồm các cột:

     - Open: Price from the first transaction of a trading day
     
     - High: Maximum price in a trading day
     
     - Low: Minimum price in a trading day
     
     - Close: Price from the last transaction of a trading day
     
     - Adj Close: Closing price adjusted to reflect the value after accounting for any corporate actions
     
     - Volume: Number of units traded in a day
     
 [Link dataset](https://www.kaggle.com/datasets/varpit94/bitcoin-data-updated-till-26jun2021), dataset này cũng có sẵn trong python dùng 'import yfinance'
 để lấy dataset.
 
 Với dataset này thì sẽ dùng các mô hình time series để dự đoán chúng , trong bài này sẽ dùng 3 mô hình đó : ARIMA, LSTM, Linear Regression
 
 # II. DATA CLEANING AND VISUALIZE
 
   Đầu tiên chúng ta kiểm tra xem có cột nào dữ liệu bị NAN hay không.
   ```php
   df.isnull().sum()
   ```
   output:
   
   ![](https://scontent.fsgn2-5.fna.fbcdn.net/v/t1.15752-9/313457409_1767206376981273_6702053681674880099_n.png?_nc_cat=106&ccb=1-7&_nc_sid=ae9488&_nc_ohc=BBfmdA5T1qsAX_KaUiE&_nc_ht=scontent.fsgn2-5.fna&oh=03_AdR2QStrm2fwrLbyCUrSSVJE9kVuitkUV_8zvqAfSlbblg&oe=63947F3C)
   
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



   

