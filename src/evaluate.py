import torch
from dataset import TimeSeriesDataset
from model import Seq2Seq
import config
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

dataset=TimeSeriesDataset("data/sample_multivariate.csv",config.SEQ_LEN,config.PRED_LEN)
model=Seq2Seq(dataset.data.shape[1],config.HIDDEN_SIZE,1,config.NUM_LAYERS,config.PRED_LEN)
model.load_state_dict(torch.load("model.pth",map_location=config.DEVICE))
model.eval()

x,y=dataset[0]
pred=model(x.unsqueeze(0)).detach().numpy().flatten()
y=y.numpy()

print("MAE",mean_absolute_error(y,pred))
print("RMSE",np.sqrt(mean_squared_error(y,pred)))
print("MAPE",np.mean(np.abs((y-pred)/y))*100)
