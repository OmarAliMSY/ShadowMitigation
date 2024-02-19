
import torch.nn as nn
import torch
import numpy as np

class LSTM_MLP(nn.Module):

    def __init__(self,input_dim,hidden_dim,layer_dim,output_dim,
                 device,bidirectional):

        super(LSTM_MLP, self).__init__()


        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.lstm1 = nn.LSTM(input_dim,hidden_dim,layer_dim,
                             batch_first=True,
                             bidirectional=bidirectional)

        self.lin0 = nn.Linear(in_features=int(hidden_dim),out_features=3)





    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        return h0, c0

    def forward(self,batch):
        h0,c0  = self.init_hidden(x=batch)

        out,(h,c) = self.lstm1(batch,(h0.detach(),c0.detach()))
        out = self.lin0(out[:,-1,:])



        return out


def lstm_mlp_train(log, config ,train_loader, valid_loader, model, optimizer,criterion,scheduler=None):

    device = config.device
    best_epoch = -1
    best_loss = 100
    state =None
    for epoch in log.t:

        tl = []
        # print(epoch)
        for i, data in enumerate(train_loader, 0):
            # event,vals = window.read(timeout=1)

            points, target = data['sequence'].reshape(-1, config.sample_size , 21 * 3).float(), data['target']
            points, target = points.to(device=device), target.to(device=device)
            optimizer.zero_grad()
            model = model.train()
            out = model(points)


            # print(out, target)
            loss = criterion(out, target.float())
            tl.append(loss.item())


            loss.backward()
            optimizer.step()

            del points
            del target
            del out


        tl = np.mean(tl)

        vl, acc = lstm_mlp_valid(config=config, valid_loader=valid_loader, model=model, criterion=criterion)
        log.updateValues(tL=tl,
                         vL=vl,
                         acc=acc
                         )
        log.updateBarText(epoch=epoch, train_loss=tl, valid_loss=vl, accuracy=acc, best_epoch=0)
        if vl < best_loss:
            best_epoch = epoch
            best_loss = vl

            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': vl,
                'train_loss': tl
            }

        if scheduler:

            scheduler.step()

    return model, optimizer, state


def lstm_mlp_valid(valid_loader, model, criterion, config):
    with torch.no_grad():
        acc = []
        vl = []
        model.eval()

        for i, data in enumerate(valid_loader, 0):
            points, target = data['sequence'].reshape(-1, config.sample_size , 21 * 3).float(), data['target']
            points, target = points.to(device=config.device), target.to(device=config.device)

            model = model.eval()
            pred = model(points)
            loss = criterion(pred, target)

            tmp_acc = torch.mean(torch.linalg.norm(pred-target,dim=1))
            acc.append(tmp_acc.cpu())

            # print(tmp_acc)
            vl.append(float(loss.item()))

            del points
            del target
            del pred

        acc = np.mean(acc)
        vl = np.mean(vl)

        return vl, acc
 