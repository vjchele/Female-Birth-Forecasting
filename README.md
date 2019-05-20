# Timeseries Forecasting Model for Female Births 

```python
import pandas as pd
from pandas import Series
from pandas import TimeGrouper
from pandas.plotting import autocorrelation_plot
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AR
```

### Load Dataset (daily-total-female-births.csv)


```python
#Load the Dataset
df = pd.read_csv('daily-total-female-births.csv',header=0, parse_dates=[0],index_col=0, squeeze=True)

```


```python
# Let's take a peek at the data
df.head()
df.tail()
```




    Date
    1959-12-27    37
    1959-12-28    52
    1959-12-29    48
    1959-12-30    55
    1959-12-31    50
    Name: Births, dtype: int64




```python
#Histogram of the data
df.hist(bins=10)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x281b5692940>




![png](Female%20Birth%20Forecasting_files/Female%20Birth%20Forecasting_5_1.png)



```python
#describing the data
df.describe()
# There are 365 records
# min birth rate per day is 23
# max birth rate per day is 73
```




    count    365.000000
    mean      41.980822
    std        7.348257
    min       23.000000
    25%       37.000000
    50%       42.000000
    75%       46.000000
    max       73.000000
    Name: Births, dtype: float64




```python
#Number of Observations
df.size
```




    365




```python
#Extracting Birth Data for the month of January 1950
df['1959-01']

```




    Date
    1959-01-01    35
    1959-01-02    32
    1959-01-03    30
    1959-01-04    31
    1959-01-05    44
    1959-01-06    29
    1959-01-07    45
    1959-01-08    43
    1959-01-09    38
    1959-01-10    27
    1959-01-11    38
    1959-01-12    33
    1959-01-13    55
    1959-01-14    47
    1959-01-15    45
    1959-01-16    37
    1959-01-17    50
    1959-01-18    43
    1959-01-19    41
    1959-01-20    52
    1959-01-21    34
    1959-01-22    53
    1959-01-23    39
    1959-01-24    32
    1959-01-25    37
    1959-01-26    43
    1959-01-27    39
    1959-01-28    35
    1959-01-29    44
    1959-01-30    38
    1959-01-31    24
    Name: Births, dtype: int64



# Visualizing Data


```python
#timeseries plot
df.plot(figsize=(15,6))
plt.show()

```


![png](Female%20Birth%20Forecasting_files/Female%20Birth%20Forecasting_10_0.png)



```python
# Dot plot of the data
df.plot(style='k.')
plt.show()
```


![png](Female%20Birth%20Forecasting_files/Female%20Birth%20Forecasting_11_0.png)



```python
df.resample('M').plot()
plt.show()

```


![png](Female%20Birth%20Forecasting_files/Female%20Birth%20Forecasting_12_0.png)



```python
#Denisty plot
df.plot(kind='kde')
plt.show()
```


![png](Female%20Birth%20Forecasting_files/Female%20Birth%20Forecasting_13_0.png)


# Baseline Forecasting Model
#### Baseline in forecast provides a point of comparison. Persistence Algorithm is a common Baseline algorithm.
#### We will use the following steps to perform baseline
#### 1.Transform the univariate dataset into a supervised learning problem.
#### 2. Establish the train and test datasets for the test harness.
#### 3. Define the persistence model.
#### 4. Make a forecast and establish a baseline performance.
#### 5. Review the complete example and plot the output.


```python
values = pd.DataFrame(df.values)
lagged_ds = pd.concat([values.shift(1),values],axis=1)
lagged_ds.columns = ['t-1','t']
```

### Train and Test Dataset


```python
# split into train and test sets
X = lagged_ds.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
```

## Persistence Algorithm


```python
# persistence model
def model_persistence(x):
    return x
# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test Mean Squared Error: %.3f' % test_score)
```

    Test Mean Squared Error: 83.744
    


```python
# Fit regression model
train_X = train_X.reshape(-1,1)
train_y = train_y.reshape(-1,1)
test_X = test_X.reshape(-1,1)

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(train_X, train_y)
regr_2.fit(train_X, train_y)
y_1 = regr_1.predict(test_X)
y_2 = regr_2.predict(test_X)
test_score = mean_squared_error(y_1, test_y)
print('Test Mean Squared Error with max_depth 2: %.3f' % test_score)
test_score = mean_squared_error(y_2, test_y)
print('Test Mean Squared Error with max_depth 5: %.3f' % test_score)


```

    Test Mean Squared Error with max_depth 2: 70.071
    Test Mean Squared Error with max_depth 5: 76.686
    

## Plot the Baseline Prediction


```python
plt.plot(train_y)
plt.plot([None for i in train_y] + [x for x in test_y])

plt.plot([None for i in train_y] + [x for x in predictions])
plt.plot([None for i in train_y] + [x for x in y_2])

plt.show()
```


![png](Female%20Birth%20Forecasting_files/Female%20Birth%20Forecasting_22_0.png)


# Autocorrelation of Data
## We can check to see if there is an autocorrelation in the data


```python
lag_plot(df)
plt.show()
# We see strong correlation only in the center and deteriorates for higher birth rates
```


![png](Female%20Birth%20Forecasting_files/Female%20Birth%20Forecasting_24_0.png)



```python
autocorrelation_plot(df)
plt.show()
```


![png](Female%20Birth%20Forecasting_files/Female%20Birth%20Forecasting_25_0.png)



```python
plot_acf(df, lags=31)
plt.show()
# Not much correlation in the data
```


![png](Female%20Birth%20Forecasting_files/Female%20Birth%20Forecasting_26_0.png)


# ARIMA Forecasting


```python
model = ARIMA(df, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())
```

    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      % freq, ValueWarning)
    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      % freq, ValueWarning)
    

                                 ARIMA Model Results                              
    ==============================================================================
    Dep. Variable:               D.Births   No. Observations:                  364
    Model:                 ARIMA(5, 1, 0)   Log Likelihood               -1245.037
    Method:                       css-mle   S.D. of innovations              7.392
    Date:                Fri, 17 May 2019   AIC                           2504.073
    Time:                        19:23:04   BIC                           2531.353
    Sample:                    01-02-1959   HQIC                          2514.916
                             - 12-31-1959                                         
    ==================================================================================
                         coef    std err          z      P>|z|      [0.025      0.975]
    ----------------------------------------------------------------------------------
    const              0.0434      0.125      0.348      0.728      -0.201       0.288
    ar.L1.D.Births    -0.7240      0.052    -13.976      0.000      -0.826      -0.622
    ar.L2.D.Births    -0.5430      0.063     -8.676      0.000      -0.666      -0.420
    ar.L3.D.Births    -0.4119      0.065     -6.309      0.000      -0.540      -0.284
    ar.L4.D.Births    -0.2876      0.063     -4.572      0.000      -0.411      -0.164
    ar.L5.D.Births    -0.1505      0.052     -2.883      0.004      -0.253      -0.048
                                        Roots                                    
    =============================================================================
                      Real          Imaginary           Modulus         Frequency
    -----------------------------------------------------------------------------
    AR.1            0.6228           -1.2380j            1.3858           -0.1758
    AR.2            0.6228           +1.2380j            1.3858            0.1758
    AR.3           -1.5311           -0.0000j            1.5311           -0.5000
    AR.4           -0.8127           -1.2645j            1.5031           -0.3409
    AR.5           -0.8127           +1.2645j            1.5031            0.3409
    -----------------------------------------------------------------------------
    


![png](Female%20Birth%20Forecasting_files/Female%20Birth%20Forecasting_28_2.png)



![png](Female%20Birth%20Forecasting_files/Female%20Birth%20Forecasting_28_3.png)


                    0
    count  364.000000
    mean    -0.006509
    std      7.404761
    min    -21.067804
    25%     -5.392352
    50%     -0.899123
    75%      4.803230
    max     25.769530
    

# Rolling Forecast AR model


```python
X = df.values

train, test = X[1:len(X)-7], X[len(X)-7:]
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
```

    Lag: 16
    Coefficients: [ 1.73118908e+01  1.55227991e-01  7.57354544e-02  5.14892409e-02
      1.83019314e-02  6.30283715e-02  2.57549513e-03  1.58567997e-01
      4.60325743e-02 -3.75935366e-02 -1.04615261e-02  3.66391301e-02
     -7.88381410e-02 -1.12589294e-02  2.79571383e-02  6.12391893e-02
      2.95111491e-02]
    predicted=41.224461, expected=44.000000
    predicted=40.971893, expected=34.000000
    predicted=40.743373, expected=37.000000
    predicted=42.067430, expected=52.000000
    predicted=42.232141, expected=48.000000
    predicted=42.116348, expected=55.000000
    predicted=41.933617, expected=50.000000
    Test MSE: 61.900
    


![png](Female%20Birth%20Forecasting_files/Female%20Birth%20Forecasting_30_1.png)


# Rolling Forecast ARIMA model


```python
X = df.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(2,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
```

    predicted=43.908002, expected=51.000000
    predicted=45.575063, expected=41.000000
    predicted=44.488854, expected=44.000000
    predicted=45.062429, expected=38.000000
    predicted=40.758497, expected=68.000000
    predicted=51.975879, expected=40.000000
    predicted=49.021711, expected=42.000000
    predicted=48.781811, expected=51.000000
    predicted=45.051095, expected=44.000000
    predicted=45.787394, expected=45.000000
    predicted=46.489904, expected=36.000000
    predicted=41.239361, expected=57.000000
    predicted=46.757501, expected=44.000000
    predicted=46.143023, expected=42.000000
    predicted=47.080515, expected=53.000000
    predicted=46.876827, expected=42.000000
    predicted=45.705319, expected=34.000000
    predicted=42.281165, expected=40.000000
    predicted=38.704266, expected=56.000000
    predicted=44.451792, expected=44.000000
    predicted=46.668603, expected=53.000000
    predicted=51.148658, expected=55.000000
    predicted=51.211879, expected=39.000000
    predicted=48.417640, expected=59.000000
    predicted=51.501405, expected=55.000000
    predicted=51.446632, expected=73.000000
    predicted=63.215640, expected=55.000000
    predicted=61.240227, expected=44.000000
    predicted=56.353159, expected=43.000000
    predicted=46.818115, expected=40.000000
    predicted=42.149542, expected=47.000000
    predicted=43.687518, expected=51.000000
    predicted=46.726705, expected=56.000000
    predicted=52.034148, expected=49.000000
    predicted=51.938188, expected=54.000000
    predicted=53.073593, expected=56.000000
    predicted=53.551790, expected=47.000000
    predicted=51.976689, expected=44.000000
    predicted=48.400860, expected=43.000000
    predicted=44.497750, expected=42.000000
    predicted=42.935045, expected=45.000000
    predicted=43.557000, expected=50.000000
    predicted=46.286581, expected=48.000000
    predicted=47.911172, expected=43.000000
    predicted=46.610998, expected=40.000000
    predicted=43.213754, expected=59.000000
    predicted=48.603981, expected=41.000000
    predicted=46.594296, expected=42.000000
    predicted=46.595714, expected=51.000000
    predicted=45.367655, expected=49.000000
    predicted=47.715724, expected=45.000000
    predicted=48.077036, expected=43.000000
    predicted=45.426044, expected=42.000000
    predicted=43.235599, expected=38.000000
    predicted=40.728971, expected=47.000000
    predicted=42.803116, expected=38.000000
    predicted=40.889630, expected=36.000000
    predicted=39.830018, expected=42.000000
    predicted=39.004321, expected=35.000000
    predicted=37.512466, expected=28.000000
    predicted=34.238577, expected=44.000000
    predicted=36.396847, expected=36.000000
    predicted=36.149167, expected=45.000000
    predicted=41.918251, expected=46.000000
    predicted=42.842770, expected=48.000000
    predicted=46.587646, expected=49.000000
    predicted=47.911781, expected=43.000000
    predicted=46.415059, expected=42.000000
    predicted=44.398617, expected=59.000000
    predicted=49.101457, expected=45.000000
    predicted=48.617977, expected=52.000000
    predicted=51.938741, expected=46.000000
    predicted=47.720264, expected=42.000000
    predicted=46.254852, expected=40.000000
    predicted=42.420349, expected=40.000000
    predicted=40.618997, expected=45.000000
    predicted=42.020354, expected=35.000000
    predicted=39.675927, expected=35.000000
    predicted=37.913834, expected=40.000000
    predicted=36.987847, expected=39.000000
    predicted=38.188708, expected=33.000000
    predicted=36.952577, expected=42.000000
    predicted=38.303688, expected=47.000000
    predicted=41.390424, expected=51.000000
    predicted=47.238791, expected=44.000000
    predicted=47.147321, expected=40.000000
    predicted=44.487529, expected=57.000000
    predicted=47.971079, expected=49.000000
    predicted=48.942360, expected=45.000000
    predicted=49.878209, expected=49.000000
    predicted=47.847955, expected=51.000000
    predicted=48.699109, expected=46.000000
    predicted=48.517149, expected=44.000000
    predicted=46.764045, expected=52.000000
    predicted=47.841457, expected=45.000000
    predicted=46.935257, expected=32.000000
    predicted=41.996117, expected=46.000000
    predicted=41.499959, expected=41.000000
    predicted=39.829040, expected=34.000000
    predicted=39.812669, expected=33.000000
    predicted=35.741177, expected=36.000000
    predicted=34.498588, expected=49.000000
    predicted=40.299481, expected=43.000000
    predicted=42.728899, expected=43.000000
    predicted=44.876520, expected=34.000000
    predicted=39.428076, expected=39.000000
    predicted=38.726147, expected=35.000000
    predicted=35.921589, expected=52.000000
    predicted=43.028513, expected=47.000000
    predicted=45.034505, expected=52.000000
    predicted=50.562460, expected=39.000000
    predicted=45.396524, expected=40.000000
    predicted=43.218497, expected=42.000000
    predicted=40.558858, expected=42.000000
    predicted=41.466487, expected=53.000000
    predicted=46.524140, expected=39.000000
    predicted=44.257066, expected=40.000000
    predicted=43.517344, expected=38.000000
    predicted=38.934284, expected=44.000000
    predicted=41.038205, expected=34.000000
    predicted=38.278900, expected=37.000000
    predicted=38.105539, expected=52.000000
    predicted=42.198460, expected=48.000000
    predicted=46.096524, expected=55.000000
    predicted=52.085076, expected=50.000000
    Test MSE: 59.487
    


![png](Female%20Birth%20Forecasting_files/Female%20Birth%20Forecasting_32_1.png)

