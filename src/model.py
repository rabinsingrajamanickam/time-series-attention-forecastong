import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size*2, hidden_size)
        self.v = nn.Linear(hidden_size,1,bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden[-1].unsqueeze(1).repeat(1,seq_len,1)
        energy = torch.tanh(self.attn(torch.cat((hidden,encoder_outputs),dim=2)))
        weights = torch.softmax(self.v(energy).squeeze(-1),dim=1)
        context = torch.bmm(weights.unsqueeze(1),encoder_outputs)
        return context, weights

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(hidden_size+1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attn = Attention(hidden_size)

    def forward(self, y_prev, hidden, cell, encoder_outputs):
        context,_ = self.attn(hidden,encoder_outputs)
        lstm_input = torch.cat((y_prev.unsqueeze(1),context),dim=2)
        output,(hidden,cell)=self.lstm(lstm_input,(hidden,cell))
        pred=self.fc(output.squeeze(1))
        return pred,hidden,cell

class Seq2Seq(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers,pred_len):
        super().__init__()
        self.encoder=Encoder(input_size,hidden_size,num_layers)
        self.decoder=Decoder(hidden_size,output_size,num_layers)
        self.pred_len=pred_len

    def forward(self,x):
        enc_out,hidden,cell=self.encoder(x)
        y=torch.zeros(x.size(0),1).to(x.device)
        outputs=[]
        for _ in range(self.pred_len):
            y,hidden,cell=self.decoder(y,hidden,cell,enc_out)
            outputs.append(y)
        return torch.stack(outputs,dim=1).squeeze(-1)
