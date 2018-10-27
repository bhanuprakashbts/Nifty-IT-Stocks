# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 15:32:27 2018

@author: Srinivas
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import nsepy
from datetime import date
from nsepy import get_history
infy = get_history(symbol='INFY',
                   start=date(2017,1,1),
                   end=date(2018,10,25))#change date as per requirement

#model building
x_close = infy.Close.values

X_train = []
y_train = []
for i in range(10, infy.shape[0]):
    X_train.append(x_close[i-10:i])
    y_train.append(x_close[i])

X_train, y_train = np.array(X_train), np.array(y_train)

infy_train_x=X_train[:-30]
infy_train_y=y_train[:-30]

infy_test_x=X_train[-30:]
infy_test_y=y_train[-30:]

#Final Model
infy_grid_lm = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
infy_grid_lm.fit(infy_train_x,infy_train_y)
infy_grid_pred=infy_grid_lm.predict(infy_test_x)


from sklearn.metrics import mean_squared_error
print(mean_squared_error(infy_test_y,infy_grid_pred))

# Visualising the results
import matplotlib.pyplot as plt
plt.plot(infy_test_y, color = 'red', label = 'Real Infosys Stock Price')
plt.plot(infy_grid_pred, color = 'blue', label = 'Predicted Infosys Stock Price')
plt.title('Infosys Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Infosys Stock Price')
plt.legend()
plt.show()


last_days = pd.DataFrame(np.array(infy.Close[-10:].values))
last_days = np.transpose(last_days)
print("Price for Tomorrow is:")
print(infy_grid_lm.predict(last_days))