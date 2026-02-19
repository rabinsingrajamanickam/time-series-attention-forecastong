import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

df=pd.read_csv("data/sample_multivariate.csv")
series=df.iloc[:,0]
train=series[:-12]
test=series[-12:]
model=ARIMA(train,order=(5,1,0)).fit()
pred=model.forecast(12)
print("Baseline MAE:",mean_absolute_error(test,pred))
