from torch import nn
from FedUtils.models.utils import Flops, FSGM
import torch
import sys
from FedUtils.models.losses import NTD_Loss
from loguru import logger


class Reshape(nn.Module):
    def forward(self, x):
        return x.reshape(-1, 576)


class Model(nn.Module):
    def __init__(self, num_classes, optimizer=None, learning_rate=None, seed=1, p_iters=10, ps_eta=0.1, pt_eta=0.001):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.num_inp = 784
        torch.manual_seed(123+seed)
        self.ntd = NTD_Loss(num_classes=num_classes)

        self.net = nn.Sequential(*[nn.Conv2d(1, 32, 5), nn.ReLU(), nn.Conv2d(32, 32, 5), nn.MaxPool2d(2), nn.ReLU(), nn.Conv2d(32, 64, 5),
                                 nn.MaxPool2d(2), nn.ReLU(), Reshape()])
        self.bottleneck = nn.Sequential(*[nn.Linear(576, 128), nn.ReLU()])
        self.head = nn.Sequential(*[nn.Linear(128, self.num_classes)])       
        self.softmax = nn.Softmax(-1)
        self.size = sys.getsizeof(self.state_dict())

        if optimizer is not None:
            self.optimizer = optimizer( list(self.net.parameters()) + list(self.bottleneck.parameters()) + list(self.head.parameters()))
        else:
            assert learning_rate, "should provide at least one of optimizer and learning rate"
            self.learning_rate = learning_rate

        self.p_iters = p_iters
        self.ps_eta = ps_eta
        self.pt_eta = pt_eta

        self.flop = Flops(self, torch.tensor([[0.0 for _ in range(self.num_inp)]]))
        if torch.cuda.device_count() > 0:
            self.net = self.net.cuda()
            self.head = self.head.cuda()
            self.bottleneck = self.bottleneck.cuda()

    def set_param(self, state_dict):
        self.load_state_dict(state_dict)
        return True

    def get_param(self):
        return self.state_dict()

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.softmax(self.forward(x))
        
    def get_classifier(self):
        odict = self.head.state_dict()
        values_list = [value for value in odict.values()]
        return torch.cat((values_list[0], values_list[1].unsqueeze(1)), dim=1)

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
    
    def loss_NTD(self, pred, gt, global_pred):
        pred = self.softmax(pred)
        global_pred = self.softmax(global_pred)
        if gt.device != pred.device:
            gt = gt.to(pred.device)
        loss = self.ntd(pred, gt, global_pred)
        return loss

    def forward(self, data):
        if data.device != next(self.parameters()).device:
            data = data.to(next(self.parameters()).device)
        data = data.reshape(-1, 1, 28, 28)
        out = self.net(data)
        out = self.bottleneck(out)
        out = self.head(out)
        return out
    
    def forward_representation(self, data):
        if data.device != next(self.parameters()).device:
            data = data.to(next(self.parameters()).device)
        data = data.reshape(-1, 1, 28, 28)
        out = self.net(data)
        out = self.bottleneck(out)
        return out

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
            func = step_func(self, data)

        for _ in range(num_epochs):
            train_iters = []
            for train_loader in data:
                train_iters.append(iter(train_loader))
            for step in range(len(train_iters[0])):
                
                for i, train_iter in enumerate(train_iters[:1]):
                    try:
                        x, y = next(train_iter)

                        c = func([x, y])
                        comp += c
                        steps += 1.0
                    except Exception as e:
                        print(e)
                        pass
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
    
    def get_representation(self, data):
        self.eval()
        representations = None
        for d in data:
            x, y = d
            with torch.no_grad():
                output = self.forward_representation(x)
                if representations is None:
                    representations = output.reshape(output.shape[0],-1)
                else:
                    representations = torch.cat((representations, output.reshape(output.shape[0],-1)), 0) 
        return representations