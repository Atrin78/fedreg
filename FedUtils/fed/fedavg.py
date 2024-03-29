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

warmup=0
data_size = 20
full = 20

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



def step_func(model, data):
    lr = model.learning_rate
    parameters = list(model.net.parameters()) + list(model.head.parameters())
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
        total_norm=0
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
            total_norm += torch.norm(lr*g)**2
  #      print('total norm')
  #      print(total_norm)
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


class FedAvg(Server):
    step = 0

    def train(self):
        indices, selected_clients = self.select_clients(0, num_clients=4)
        active_clients = np.random.choice(selected_clients, round(4*(1.0-self.drop_percent)), replace=False)
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
                if r< warmup:
                    stats = self.testAE()
                else:
                    stats = self.test()
                if self.eval_train:
                    if r<warmup:
                        stats_train = self.train_error_and_lossAE()
                    else:
                        stats_train = self.train_error_and_loss()
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

            transform_cifar = transforms.Compose(
            [
             torchvision.transforms.functional.rgb_to_grayscale,
             transforms.ToTensor(),
             torchvision.transforms.Resize(28),
             ])
            cifar = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=transform_cifar)
            cifar = torch.utils.data.Subset(cifar, list(np.random.choice(len(cifar), data_size)))
            cifar_data, cifar_y = next(iter(DataLoader(cifar, batch_size=len(cifar))))
            cifar_data = torch.reshape(cifar_data, (-1, 784))
            cifar_mean = torch.mean(cifar_data).item()
            cifar_std = torch.std(cifar_data).item()
            cifar_data = (cifar_data - cifar_mean) / (cifar_std + 0.00001)
            cifar = torch.utils.data.TensorDataset(cifar_data, cifar_y)

            


            for idx, c in enumerate(active_clients):
                c.set_param(self.model.get_param())
                #if idx==0:
                coef=1
                #else:
                #    coef=0
                if r < warmup:
                    soln, stats = c.solve_inner(num_epochs=self.num_epochs*2, step_func=step_func2, coef=coef)  # stats has (byte w, comp, byte r)
                elif r < full:
                    soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=step_func, coef=coef)  # stats has (byte w, comp, byte r)
                else:
                    c.gen_data = cifar
                    soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=(step_func4, step_func5), coef=coef)  # stats has (byte w, comp, byte r)
                soln = [1.0, soln[1]]
                w += soln[0]
                if len(csolns) == 0:
                    csolns = {x: soln[1][x].detach()*soln[0] for x in soln[1]}
                    print(c.id)
               #     norm = 0
               #     for x in csolns:
               #         norm += torch.norm(soln[1][x].detach().cpu()-self.model.get_param()[x].detach().cpu())**2
               #     print('norm')
               #     print(norm)
                else:
                    print(c.id)
               #     norm = 0
               #     for x in csolns:
               #         norm += torch.norm(soln[1][x].detach().cpu()-self.model.get_param()[x].detach().cpu())**2
               #     print('norm')
               #     print(norm)
               #     for x in csolns:
               #         csolns[x].data.add_(soln[1][x]*soln[0])
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
