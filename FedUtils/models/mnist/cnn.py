from torch import nn
from FedUtils.models.utils import Flops, FSGM
import torch
import sys

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0.01)


class Reshape(nn.Module):
    def forward(self, x):
        return x.reshape(-1, 1024)

class ReverseReshape(nn.Module):
    def forward(self, x):
        return x.reshape(-1, 64, 4, 4)

class Model(nn.Module):
    def __init__(self, num_classes, optimizer=None, learning_rate=None, seed=1, p_iters=10, ps_eta=0.1, pt_eta=0.001):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.num_inp = 1024
        torch.manual_seed(123+seed)

        self.net = nn.Sequential(*[nn.Conv2d(1, 32, 5), nn.ReLU(), nn.Conv2d(32, 32, 5), nn.MaxPool2d(2), nn.ReLU(), nn.Conv2d(32, 64, 5),
                                 nn.MaxPool2d(2), nn.ReLU(), Reshape(), nn.Linear(1024, 256), nn.ReLU()])
        self.head = nn.Linear(256, self.num_classes)
        self.decoder = nn.Sequential(*[nn.Linear(256, 1024), ReverseReshape(), nn.ConvTranspose2d(64, 32, 5, padding=2), nn.Upsample(scale_factor=2), nn.ReLU(), nn.ConvTranspose2d(32, 32, 5, padding=2), nn.Upsample(scale_factor=2), nn.ReLU(), nn.ConvTranspose2d(32, 1, 5, padding=2), nn.Upsample(scale_factor=2), nn.Sigmoid()])
        self.size = sys.getsizeof(self.state_dict())
        self.softmax = nn.Softmax(-1)
        mm=1
        for i in range(mm):
            self.net.apply(init_weights)


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
            self.net = self.net.cuda()
            self.head = self.head.cuda()
            self.decoder = self.decoder.cuda()

    def set_param(self, state_dict):
        self.load_state_dict(state_dict)
        return True

    def get_param(self):
        return self.state_dict()

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.softmax(self.forward(x))

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

    def MSE(self, pred, gt):
        if gt.device != pred.device:
            gt = gt.to(pred.device)
        loss = (pred - gt)**2
     #   print(loss[0][0][:10,:10])
     #   print(loss.max())
     #   print(loss.shape)
        loss = loss.sum([-1, -2, -3])
        return loss


    def forward(self, data):
        if data.device != next(self.parameters()).device:
            data = data.to(next(self.parameters()).device)
        data = data.reshape(-1, 1, 32, 32)
        out = self.net(data)
        out = self.head(out)
        return out

    def AE(self, data):
        if data.device != next(self.parameters()).device:
            data = data.to(next(self.parameters()).device)
        data = data.reshape(-1, 1, 32, 32)
        out = self.net(data)
     #   out = self.head(out)
        out = self.decoder(out)
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
        data = data[0]
        comp = 0.0
        weight = 1.0
        steps = 0
        if step_func is None:
            func = self.train_onestep
        else:
            func = step_func(self, data)

        for _ in range(num_epochs):
            for x, y in data:
                c = func([x, y])
                comp += c
                steps += 1.0
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


    def testAE(self, data):
        tot_correct = 0.0
        loss = 0.0
        self.eval()
        for d in data:
            x, y = d
            with torch.no_grad():
                pred = self.AE(x)
            loss += self.MSE(pred, x).mean()
         #   pred_max = pred.argmax(-1).float()
          #  assert len(pred_max.shape) == len(y.shape)
          #  if pred_max.device != y.device:
          #      pred_max = pred_max.detach().to(y.device)
          #  tot_correct += (pred_max == y).float().sum()
        return loss, loss
