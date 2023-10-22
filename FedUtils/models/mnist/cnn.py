from torch import nn
from FedUtils.models.utils import Flops, FSGM
import torch
import sys
import copy
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0.01)


class Reshape(nn.Module):
    def forward(self, x):
        return x.reshape(-1, 576)

class ReverseReshape(nn.Module):
    def forward(self, x):
        return x.reshape(-1, 64, 4, 4)

def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits  
class NTD_Loss(nn.Module):
    """Not-true Distillation Loss"""

    def __init__(self, num_classes=10, tau=3, beta=1):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta

    def forward(self, logits, targets, dg_logits):
        ce_loss = self.CE(logits, targets)
        ntd_loss = self._ntd_loss(logits, dg_logits, targets)

        loss = ce_loss + self.beta * ntd_loss

        return loss

    def _ntd_loss(self, logits, dg_logits, targets):
        """Not-tue Distillation Loss"""

        # Get smoothed local model prediction
        logits = refine_as_not_true(logits, targets, self.num_classes)
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau ** 2) * self.KLDiv(pred_probs, dg_probs)

        return loss


class FedDecorrLoss(nn.Module):

    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        N, C = x.shape
        if N == 1:
            return 0.0

        x = x - x.mean(dim=0, keepdim=True)
        x = x / (self.eps + x.std(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)/(N-1)

       # print(N)
       # print(corr_mat[:5, :5])

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        #loss = loss / N


        return loss.mean()

class Model(nn.Module):
    def __init__(self, num_classes, optimizer=None, learning_rate=None, seed=1, p_iters=10, ps_eta=0.1, pt_eta=0.001):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.num_inp = 784
        torch.manual_seed(123+seed)

        self.decorr = FedDecorrLoss()
        self.adapt = nn.BatchNorm2d(1)
        self.ntd = NTD_Loss(num_classes=num_classes)
        self.net = nn.Sequential(*[nn.Conv2d(1, 32, 5), nn.ReLU(), nn.Conv2d(32, 32, 5), nn.MaxPool2d(2), nn.ReLU(), nn.Conv2d(32, 64, 5),
                                 nn.MaxPool2d(2), nn.ReLU(), Reshape()])
        self.bottleneck = nn.Sequential(*[nn.Linear(576, 128), nn.ReLU()])
        self.head = nn.Sequential(*[nn.Linear(128, self.num_classes)])
        self.decoder = nn.Sequential(*[nn.Linear(128, 1024), ReverseReshape(), nn.Upsample(scale_factor=2), nn.ConvTranspose2d(64, 32, 5, padding=2), nn.ReLU(), nn.Upsample(scale_factor=2), nn.ConvTranspose2d(32, 32, 5, padding=2), nn.ReLU(), nn.Upsample(scale_factor=2), nn.ConvTranspose2d(32, 1, 5, padding=2), nn.Sigmoid()])
        self.size = sys.getsizeof(self.state_dict())
        self.softmax = nn.Softmax(-1)
        self.global_model = None
      #  mm=1
      #  for i in range(mm):
      #      self.net.apply(init_weights)


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
            self.adapt = self.adapt.cuda()
            self.net = self.net.cuda()
            self.head = self.head.cuda()
            self.decoder = self.decoder.cuda()
            self.bottleneck = self.bottleneck.cuda()

    def set_param(self, state_dict):
        self.load_state_dict(state_dict, strict=False)
        if self.global_model is not None:
            if torch.cuda.device_count() > 0:
                self.global_model.cuda()
            for params in self.global_model.parameters():
                params.requires_grad = False
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
    
    def loss_NTD(self, pred, gt, global_pred):
        pred = self.softmax(pred)
        global_pred = self.softmax(global_pred)
        if gt.device != pred.device:
            gt = gt.to(pred.device)
        loss = self.ntd(pred, gt, global_pred)
        return loss

    def MSE(self, pred, gt):
        if gt.device != pred.device:
            gt = gt.to(pred.device)
        loss = nn.MSELoss()(pred, gt)
     #   loss = (pred - gt)**2
     #   print(loss[0][0][:10,:10])
     #   print(loss.max())
     #   print(loss.shape)
     #   loss = loss.sum([-1, -2, -3])
        return loss


    def forward(self, data):
        if data.device != next(self.parameters()).device:
            data = data.to(next(self.parameters()).device)
    #    data_min = torch.transpose(torch.min(data, 1)[0].repeat((784, 1)),0, 1)
    #    data_max = torch.transpose(torch.max(data, 1)[0].repeat((784, 1)),0, 1)
    #    data = (data - data_min)/(data_max-data_min)
        data = data.reshape(-1, 1, 28, 28)
        out = self.net(data)
        out = self.bottleneck(out)
        out = self.head(out)
        return out

    def forward_adapt(self, data):
        if data.device != next(self.parameters()).device:
            data = data.to(next(self.parameters()).device)
    #    data_min = torch.transpose(torch.min(data, 1)[0].repeat((784, 1)),0, 1)
    #    data_max = torch.transpose(torch.max(data, 1)[0].repeat((784, 1)),0, 1)
    #    data = (data - data_min)/(data_max-data_min)
        data = data.reshape(-1, 1, 28, 28)
        out = self.adapt(data)
        out = self.net(out)
        out = self.bottleneck(out)
        out = self.head(out)
        return out

    def forward_decorr(self, data):
        if data.device != next(self.parameters()).device:
            data = data.to(next(self.parameters()).device)
    #    data_min = torch.transpose(torch.min(data, 1)[0].repeat((784, 1)),0, 1)
    #    data_max = torch.transpose(torch.max(data, 1)[0].repeat((784, 1)),0, 1)
    #    data = (data - data_min)/(data_max-data_min)
        data = data.reshape(-1, 1, 28, 28)
        out = self.net(data)
        features = self.bottleneck(out)
        out = self.head(features)
        return out, features


    def AE(self, data):
        if data.device != next(self.parameters()).device:
            data = data.to(next(self.parameters()).device)
    #    data_min = torch.transpose(torch.min(data, 1)[0].repeat((784, 1)),0, 1)
    #    data_max = torch.transpose(torch.max(data, 1)[0].repeat((784, 1)),0, 1)
    #    data = (data - data_min)/(data_max-data_min)
        data = data.reshape(-1, 1, 32, 32)
        out = self.net(data)
     #   out = self.head(out)
        out = self.bottleneck(out)
        out = self.decoder(out)
    #    out = out[:, :, 2:-2, 2:-2]
    #    out = torch.reshape(out, (-1, 784))
     #   print('oo')
     #   print(out)
        return out

    def multi(self, data):
        if data.device != next(self.parameters()).device:
            data = data.to(next(self.parameters()).device)
        data = data.reshape(-1, 1, 32, 32)
        out = self.net(data)
        out = self.bottleneck(out)
        rec = self.decoder(out)
        logit = self.head(out)
        return logit, rec

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

  #  def solve_inner(self, data, num_epochs=1, step_func=None):
  #      data = data[0]
  #      comp = 0.0
  #      weight = 1.0
  #      steps = 0
  #      if step_func is None:
  #          func = self.train_onestep
  #      else:
  #          func = step_func(self, data)

  #      for _ in range(num_epochs):
  #          for x, y in data:
  #              c = func([x, y])
  #              comp += c
  #              steps += 1.0
  #      soln = self.get_param()
  #      return soln, comp, weight

    def solve_inner(self, data, num_epochs=1, step_func=None):
        comp = 0.0
        weight = 1.0
        steps = 0
        if step_func is None:
            func = self.train_onestep
        else:
            if self.global_model is not None:
                func = step_func(self, data, self.global_model)
            else:
                func = step_func(self, data)

        for _ in range(num_epochs):
            train_iters = []
         #   train_w = [1.0, 0.1]
         #   if len(data)==1:
         #       train_w = [1.0]
            for train_loader in data:
                train_iters.append(iter(train_loader))
         #   aux_x,_ = next(train_iters[1])
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


    def testAE(self, data):
        tot_correct = 0.0
        loss = 0.0
        self.eval()
        for d in data:
            x, y = d
            with torch.no_grad():
                pred = self.AE(x)
         #   data = x
         #   data_min = torch.transpose(torch.min(data, 1)[0].repeat((784, 1)),0, 1)
         #   data_max = torch.transpose(torch.max(data, 1)[0].repeat((784, 1)),0, 1)
         #   data = (data - data_min)/(data_max-data_min)
            loss += self.MSE(pred, x).mean()
         #   pred_max = pred.argmax(-1).float()
          #  assert len(pred_max.shape) == len(y.shape)
          #  if pred_max.device != y.device:
          #      pred_max = pred_max.detach().to(y.device)
          #  tot_correct += (pred_max == y).float().sum()
        return loss, loss
