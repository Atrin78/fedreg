from torch import nn
from FedUtils.models.utils import Flops, FSGM
import torch
import sys
from typing import Tuple, Optional, List, Dict


class Reshape(nn.Module):
    def forward(self, x):
        return x.reshape(-1, 50176)


class Model(nn.Module):
    def __init__(self, num_classes, optimizer=None, learning_rate=None, seed=1, p_iters=10, ps_eta=0.1, pt_eta=0.001):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.num_inp = 128*128*3
        torch.manual_seed(123+seed)

        self.net = nn.Sequential(*[nn.Conv2d(3, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(64, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(64, 128, 5, 1, 2), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(128 * 16 * 16, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True), nn.Linear(2048, 512), nn.BatchNorm1d(512)])
       # self.net = nn.Sequential(*[nn.Conv2d(3, 32, 5), nn.ReLU(), nn.Conv2d(32, 32, 5), nn.MaxPool2d(2), nn.ReLU(), nn.Conv2d(32, 64, 5), nn.MaxPool2d(2), nn.ReLU(), Reshape(), nn.Linear(50176, 512), nn.ReLU()])
      #  self.net = nn.Sequential(*[nn.Conv2d(3, 64, 11, 4, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2), nn.Conv2d(64, 192, 5, 1, 2), nn.BatchNorm2d(192), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2), nn.Conv2d(192, 384, 3, 1, 1), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.Conv2d(384, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2), nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(), nn.Linear(256 * 6 * 6, 4096), nn.BatchNorm1d(4096), nn.ReLU(inplace=True), nn.Linear(4096, 4096), nn.BatchNorm1d(4096), nn.ReLU(inplace=True)])
    #    self.net = nn.Sequential(*[nn.Conv2d(1, 32, 5,padding="same"), nn.ReLU(), nn.Conv2d(32, 32, 5,padding="same"), nn.MaxPool2d(2), nn.ReLU(), nn.Conv2d(32, 64, 5, padding="same"), nn.MaxPool2d(2), nn.ReLU(), Reshape(), nn.Linear(1024, 256), nn.ReLU()])
        self.head = nn.Linear(512, self.num_classes)
  #      self.whole = nn.Sequential(*[self.net, self.head])
        self.size = sys.getsizeof(self.state_dict())
        self.softmax = nn.Softmax(-1)

        if optimizer is not None:
            self.optimizer = optimizer(self.parameters())
        else:
            assert learning_rate, "should provide at least one of optimizer and learning rate"
            self.learning_rate = learning_rate

        self.p_iters = p_iters
        self.ps_eta = ps_eta
        self.pt_eta = pt_eta

        self.flop = Flops(self, torch.tensor([[0.0 for _ in range(self.num_inp)]]))
        if torch.cuda.device_count() > 0:
   #         self = self.cuda()
            #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #self.to(device)
            self.net = self.net.cuda()
            self.head = self.head.cuda()

    def set_param(self, state_dict):
        self.load_state_dict(state_dict)
        return True

    def get_param(self):
        return self.state_dict()

 #   def get_parameters(self) -> List[Dict]:
 #       """A parameter list which decides optimization hyper-parameters,
 #           such as the relative learning rate of each layer
 #       """
 #       params = [
 #           {"params": self.net.parameters(), "lr_mult": 0.1},
 #           {"params": self.head.parameters(), "lr_mult": 1.},
 #       ]
 #       return params

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            pred = self.forward(x)
            return self.softmax(pred)

    def generate_fake(self, x, y):
        self.eval()
        psuedo, perturb = x.detach(), x.detach()
        if psuedo.device != next(self.parameters()).device:
            psuedo = psuedo.to(next(self.parameters()).device)
            perturb = perturb.to(next(self.parameters()).device)
        psuedo = FSGM(self, psuedo, y, self.p_iters, self.ps_eta)
        perturb = FSGM(self, perturb, y, self.p_iters, self.pt_eta)
        psuedo_y, perturb_y = self.predict(psuedo), self.predict(perturb)
        return [psuedo, y, psuedo_y], [perturb, y, perturb_y]

    def loss(self, pred, gt):
        pred = self.softmax(pred)
        if gt.device != pred.device:
            gt = gt.to(pred.device)
        if len(gt.shape) != len(pred.shape):
            gt = nn.functional.one_hot(gt.long(), self.num_classes).float()
        assert len(gt.shape) == len(pred.shape)
        loss = -gt*torch.log(pred+1e-12)
        loss = loss.sum(1)
        return loss

    def forward(self, data):
        if data.device != next(self.parameters()).device:
            data = data.to(next(self.parameters()).device)
        data = data.reshape(-1, 3, 128, 128)
   #     x = data
   #     for layer in self.whole:
   #         pred = x
   #         x = layer(x)
        out = self.net(data)
        pred = self.head(out)
        return pred

    def forward_emb(self, data):
        if data.device != next(self.parameters()).device:
            data = data.to(next(self.parameters()).device)
        data = data.reshape(-1, 3, 128, 128)
   #     x = data
   #     for layer in self.whole:
   #         pred = x
   #         x = layer(x)
        out = self.net(data)
        pred = self.head(out)
        return pred, out

    def train_onestep(self, data):
        self.train()
        self.zero_grad()
        self.optimizer.zero_grad()
        x, y = data
        pred = self.forward(x)
        loss = self.loss(pred, y).mean()
        loss.backward()
        self.optimizer.step()

        return self.flop*len(x)

    def solve_inner(self, data, num_epochs=1, step_func=None):
        comp = 0.0
        weight = 1.0
        steps = 0
        if step_func is None:
            func = self.train_onestep
        else:
            func = step_func(self, data[0])

        for _ in range(num_epochs):
            train_iters = []
            train_w = [1.0, 0.25]
            if len(data)==1:
                train_w = [1.0]
            for train_loader in data:
                train_iters.append(iter(train_loader))
            for step in range(len(train_iters[0])):
           #     xt, yt = None, None
           #     wt = None
                for i, train_iter in enumerate(train_iters):
                    try:
                        x, y = next(train_iter)
                        if x.shape[0]==1:
                            continue
                        
               #         w = torch.ones((y.shape[0],)).to(device)
                #        x = x.to(device)
                #        y = y.to(device)
               #         print(torch.max(x))
               #         if xt is None:
               #            xt, yt = x, y
               #           wt = w * train_w[i]
               #         else:
               #             xt = torch.cat((xt, x), 0)
               #             yt = torch.cat((yt, y), 0)
               #             wt = torch.cat((wt, w * train_w[i]), 0)
                        c = func([x, y])
                        comp += c
                        steps += 1.0
                    except Exception as e:
                        print(x.shape)
                        print(e)
                        
             #   c = func([xt, yt], wt)
             #   comp += c
             #   steps += 1.0


         #   for x, y in data:
         #       c = func([x, y])
         #       comp += c
         #       steps += 1.0
        soln = self.get_param()
        return soln, comp, weight

    def test(self, data):
        tot_correct = 0.0
        loss = 0.0
        self.eval()
        for d in data:
            x, y = d
            with torch.no_grad():
                pred = self.forward(x)
            loss += self.loss(pred, y).sum()
            pred_max = pred.argmax(-1).float()
            assert len(pred_max.shape) == len(y.shape)
            if pred_max.device != y.device:
                pred_max = pred_max.detach().to(y.device)
            tot_correct += (pred_max == y).float().sum()
        return tot_correct, loss

    def test_ind(self, data):
        tot_correct = 0.0
        loss = []
        self.eval()
        for d in data:
            x, y = d
            with torch.no_grad():
                pred = self.forward(x)
            loss.append(self.loss(pred, y))
            pred_max = pred.argmax(-1).float()
            assert len(pred_max.shape) == len(y.shape)
            if pred_max.device != y.device:
                pred_max = pred_max.detach().to(y.device)
            tot_correct += (pred_max == y).float().sum()
        return tot_correct, loss
		
		
    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.net.parameters(), "lr_mult": 0.1},
            {"params": self.head.parameters(), "lr_mult": 0.1},
        ]
        return params
