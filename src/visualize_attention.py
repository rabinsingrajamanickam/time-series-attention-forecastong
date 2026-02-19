import matplotlib.pyplot as plt
import torch
from dataset import TimeSeriesDataset
from model import Seq2Seq
import config

dataset=TimeSeriesDataset("data/sample_multivariate.csv",config.SEQ_LEN,config.PRED_LEN)
model=Seq2Seq(dataset.data.shape[1],config.HIDDEN_SIZE,1,config.NUM_LAYERS,config.PRED_LEN)
model.load_state_dict(torch.load("model.pth",map_location=config.DEVICE))
model.eval()

x,_=dataset[0]
output=model(x.unsqueeze(0))
plt.plot(output.detach().numpy().flatten())
plt.title("Prediction")
plt.show()
