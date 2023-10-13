from .server import Server
from loguru import logger
import numpy as np
from FedUtils.models.utils import decode_stat
import torch
import torchvision
import itertools
import torchvision.transforms as transforms
from FedUtils.models.utils import read_data, CusDataset, ImageDataset
from torch.utils.data import DataLoader

warmup=5
data_size = 20
full = 0
num_adapt=40

def step_func4(model, data):
    lr = model.learning_rate
    parameters = list(model.net.parameters()) + list(model.bottleneck.parameters()) + list(model.head.parameters())
    flop = model.flop

    def func(d):
        nonlocal flop, lr
        model.train()
        model.zero_grad()
        x, y, aux_x = d
    #    x = torch.reshape(torchvision.transforms.functional.rotate(torch.reshape(x, (-1, 28, 28)), np.random.uniform(-1, 1)), (-1, 784))
        pred, features = model.forward_decorr(x)
        _, aux_features = model.forward_decorr(aux_x)
        loss1 = model.loss(pred, y).mean()
        loss2 = model.decorr.forward(torch.cat((features, aux_features), 0))
        #loss2 = model.decorr.forward(features)
        #print('losses')
        #print(loss1)
        #print(loss2)
        loss=loss1+0.5*loss2
        grad = torch.autograd.grad(loss, parameters)
    #    print('g')
    #    print(grad[0][0])
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
        return flop*len(x)
    return func

def step_func5(model, data):
    lr = model.learning_rate*0.1
    parameters = list(model.net.parameters()) + list(model.bottleneck.parameters())
    flop = model.flop

    def func(d):
        nonlocal flop, lr
        model.train()
        model.zero_grad()
        x, y = d
    #    x = torch.reshape(torchvision.transforms.functional.rotate(torch.reshape(x, (-1, 28, 28)), np.random.uniform(-1, 1)), (-1, 784))
        pred, features = model.forward_decorr(x)
        loss = model.decorr.forward(features)
        grad = torch.autograd.grad(loss, parameters)
    #    print('g')
    #    print(grad[0][0])
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
        return flop*len(x)
    return func

def step_func6(model, data):
    lr = model.learning_rate
    parameters = list(model.net.parameters()) + list(model.bottleneck.parameters()) + list(model.head.parameters())
    flop = model.flop

    def func(d):
        nonlocal flop, lr
        model.train()
        model.zero_grad()
        x, y = d
    #    x = torch.reshape(torchvision.transforms.functional.rotate(torch.reshape(x, (-1, 28, 28)), np.random.uniform(-1, 1)), (-1, 784))
        pred = model.forward_adapt(x)
        loss1 = model.loss(pred, y).mean()
        #loss2 = model.decorr.forward(features)
        #print('losses')
        #print(loss1)
        #print(loss2)
        loss=loss1
        grad = torch.autograd.grad(loss, parameters)
    #    print('g')
    #    print(grad[0][0])
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
        return flop*len(x)
    return func

def step_func7(model, data):
    lr = model.learning_rate
    parameters = list(model.adapt.parameters()) + list(model.net.parameters()) + list(model.bottleneck.parameters()) + list(model.head.parameters())
    flop = model.flop

    def func(d):
        nonlocal flop, lr
        model.train()
        model.zero_grad()
        x, y = d
    #    x = torch.reshape(torchvision.transforms.functional.rotate(torch.reshape(x, (-1, 28, 28)), np.random.uniform(-1, 1)), (-1, 784))
        pred = model.forward_adapt(x)
        loss1 = model.loss(pred, y).mean()
        #loss2 = model.decorr.forward(features)
        #print('losses')
        #print(loss1)
        #print(loss2)
        loss=loss1
        grad = torch.autograd.grad(loss, parameters)
    #    print('g')
    #    print(grad[0][0])
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
        return flop*len(x)
    return func

def step_func8(model, data):
    lr = model.learning_rate
    parameters = list(model.adapt.parameters())
    flop = model.flop

    def func(d):
        nonlocal flop, lr
        model.train()
        model.zero_grad()
        x, y = d
    #    x = torch.reshape(torchvision.transforms.functional.rotate(torch.reshape(x, (-1, 28, 28)), np.random.uniform(-1, 1)), (-1, 784))
        pred = model.forward_adapt(x)
        loss1 = model.loss(pred, y).mean()
        #loss2 = model.decorr.forward(features)
        #print('losses')
        #print(loss1)
        #print(loss2)
        loss=loss1
        grad = torch.autograd.grad(loss, parameters)
    #    print('g')
    #    print(grad[0][0])
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
        return flop*len(x)
    return func


def step_func(model, data):
    lr = model.learning_rate
    parameters = list(model.head.parameters())
    flop = model.flop

    def func(d):
        nonlocal flop, lr
        model.train()
        model.zero_grad()
        x, y = d
        
        pred = model.forward(x)
        loss = model.loss(pred, y).mean()
        grad = torch.autograd.grad(loss, parameters)
    #    print('g')
    #    print(grad[0][0])
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
        return flop*len(x)
    return func


def step_func3(model, data):
    lr = model.learning_rate
#    parameters = list(model.net.parameters()) + list(model.bottleneck.parameters()) + list(model.head.parameters())
    parameters = list(model.parameters())
  #  parameters = itertools.chain(*[model.net.parameters(), model.bottleneck.parameters(), model.head.parameters()])
    flop = model.flop

    def func(d):
        nonlocal flop, lr
        model.train()
        model.zero_grad()
        x, y = d
    #    x = torch.reshape(torchvision.transforms.functional.rotate(torch.reshape(x, (-1, 28, 28)), np.random.uniform(-1, 1)), (-1, 784))
        pred, rec = model.multi(x)
        loss1 = model.loss(pred, y).mean()
        loss2 = model.MSE(rec, x).mean()
        loss=loss1+loss2
        grad = torch.autograd.grad(loss, parameters)
    #    print('g')
    #    print(grad[0][0])
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
        return flop*len(x)
    return func



def step_func2(model, data):
    lr = model.learning_rate*10
    parameters = list(model.net.parameters()) + list(model.bottleneck.parameters())  + list(model.decoder.parameters())
   # parameters = model.parameters()
#    parameters = itertools.chain(*[model.net.parameters(), model.bottleneck.parameters(), model.decoder.parameters()])
    flop = model.flop

    def func(d):
        nonlocal flop, lr
        model.train()
        model.zero_grad()
        x, y = d
     #   x = torch.reshape(torchvision.transforms.functional.rotate(torch.reshape(x, (-1, 28, 28)), np.random.uniform(-1, 1)), (-1, 784))
        noisy_x = x+(0.04**0.5)*torch.randn(x.shape)
        noisy_x = noisy_x.clamp(0.0, 1.0)
        pred = model.AE(noisy_x)
        loss = model.MSE(pred, x)
        loss = loss.mean()
        print(loss)
        grad = torch.autograd.grad(loss, parameters)
     #   print('g')
     #   print(grad[-3:])
  #      print(parameters[-3:])
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
        return flop*len(x)
    return func


class FedAdapt(Server):
    step = 0

    def train(self):
        indices, selected_clients = self.select_clients(0, num_clients=300)
        active_clients = np.random.choice(selected_clients, round(300*(1.0-self.drop_percent)), replace=False)
        xx = None
        for idx, c in enumerate(active_clients):
           x, _ = next(iter(c.train_data))
           if xx is None:
               xx = x
           else:
               torch.cat((xx, x), 0)
        x_mean = torch.mean(xx)
        x_std = torch.std(xx)
        CusDataset.x_mean = x_mean.item()
        CusDataset.x_std = x_std.item()

        logger.info("Train with {} workers...".format(self.clients_per_round))
        for r in range(self.num_rounds):
            if r % self.eval_every == 0:
                logger.info("-- Log At Round {} --".format(r))
                
                stats = self.test_adapt(step_func8, num_adapt)
                if self.eval_train:
                    stats_train = self.train_error_and_loss_adapt(step_func8, num_adapt)
                else:
                    stats_train = stats
                logger.info("-- TEST RESULTS --")
                decode_stat(stats)
                logger.info("-- TRAIN RESULTS --")
                decode_stat(stats_train)

            indices, selected_clients = self.select_clients(r, num_clients=self.clients_per_round)
            np.random.seed(r)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round*(1.0-self.drop_percent)), replace=False)
            print('active')
            print([c.id for c in active_clients])
            csolns = {}
            w = 0


            


            for idx, c in enumerate(active_clients):
                c.set_param(self.model.get_param())
                #if idx==0:
                coef=1
                #else:
                #    coef=0
                if r < warmup:
                    soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=step_func7, coef=coef)  # stats has (byte w, comp, byte r)
                else:
                    soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=step_func8, coef=coef)  # stats has (byte w, comp, byte r)
                    soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=step_func6, coef=coef)  # stats has (byte w, comp, byte r)
                soln = [1.0, soln[1]]
                w += soln[0]
                #_ = [print(soln[1][x]) for x in soln[1] if x.split('.')[0]=='adapt']
                if len(csolns) == 0:
                    csolns = {x: soln[1][x].detach()*soln[0] for x in soln[1] if x.split('.')[0] != 'adapt'}
                else:
                    for x in csolns:
                        if x.split('.')[0] != 'adapt':
                            csolns[x].data.add_(soln[1][x]*soln[0])
                del c
            csolns = [[w, {x: csolns[x]/w for x in csolns}]]

            self.latest_model = self.aggregate(csolns)

        logger.info("-- Log At Round {} --".format(r))
        stats = self.test()
        if self.eval_train:
            stats_train = self.train_error_and_loss()
        else:
            stats_train = stats
        logger.info("-- TEST RESULTS --")
        decode_stat(stats)
        logger.info("-- TRAIN RESULTS --")
        decode_stat(stats_train)
