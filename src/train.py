import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import TimeSeriesDataset
from model import Seq2Seq
import config

dataset = TimeSeriesDataset("data/sample_multivariate.csv",config.SEQ_LEN,config.PRED_LEN)
loader = DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True)

model = Seq2Seq(input_size=dataset.data.shape[1],hidden_size=config.HIDDEN_SIZE,output_size=1,num_layers=config.NUM_LAYERS,pred_len=config.PRED_LEN).to(config.DEVICE)
optimizer=torch.optim.Adam(model.parameters(),lr=config.LR)
loss_fn=nn.MSELoss()

for epoch in range(config.EPOCHS):
    total=0
    for x,y in loader:
        x,y=x.to(config.DEVICE),y.to(config.DEVICE)
        pred=model(x)
        loss=loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total+=loss.item()
    print(epoch,total/len(loader))

torch.save(model.state_dict(),"model.pth")
