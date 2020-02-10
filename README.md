# Nifty IT Stocks exploration

This workings contains
1. 14,16,....,52 week moving average(closing price) for each stock and index.
2. The following dummy time series:
   2.1 Volume shocks - If volume traded is 10% higher/lower than previous day - make a 0/1 boolean time series for shock, 0/1 dummy-coded time series for direction of shock.
   2.2 Price shocks - If closing price at T vs T+1 has a difference > 2%, then 0/1 boolean time series for shock, 0/1 dummy-coded time series for direction of shock.
   2.3 Pricing shock without volume shock - based on points 2.1 & 2.2 -  0/1 dummy time series.
 
 3.Prediction model for TCS based on comparison of RandomForest Regressor and Linear regession from sklearn.linear_model
