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
 
 Với dataset này thì sẽ dùng các mô hình time series để dự đoán chúng và bài này chỉ dự đoán biến 'ADj Close', trong bài này sẽ dùng 3 mô hình đó : ARIMA, LSTM, VAR
 
 # II. DATA CLEANING AND VISUALIZE
 
   Đầu tiên chúng ta kiểm tra xem có cột nào dữ liệu bị NAN hay không.
   ```php
   df.isnull().sum()
   ```
   output:
   
   ![](https://scontent.fsgn2-5.fna.fbcdn.net/v/t1.15752-9/313457409_1767206376981273_6702053681674880099_n.png?_nc_cat=106&ccb=1-7&_nc_sid=ae9488&_nc_ohc=BBfmdA5T1qsAX_KaUiE&_nc_ht=scontent.fsgn2-5.fna&oh=03_AdR2QStrm2fwrLbyCUrSSVJE9kVuitkUV_8zvqAfSlbblg&oe=63947F3C)
   
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

# III. Xâu dựng mô hình
Sau khi tiến hành kiểm tra và khám phá dữ liệu xong thì tiến thành chia dữ liệu thành tập train và test để xây dựng mô hình.
```php
# train and test
to_row= int(len(x)*0.8)
training_data= list(x[0: to_row])
testing_data= list(x[to_row:])
```
chúng ta dữ liệu 80/20, 80% dữ liệu dùng cho việc train model, và 20% dữ liệu chúng ta cho model dự đoán để đánh giá kết quả của model.


   

