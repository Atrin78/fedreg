from .server import Server
from loguru import logger
import numpy as np
from FedUtils.models.utils import decode_stat
import torch

warmup=10

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
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
        return flop*len(x)
    return func


def step_func2(model, data):
    lr = model.learning_rate
    parameters = list(model.net.parameters()) + list(model.decoder.parameters())
    flop = model.flop

    def func(d):
        nonlocal flop, lr
        model.train()
        model.zero_grad()
        x, y = d
        print(x.shape)
        pred = model.AE(x)
        loss = model.MSE(pred, x+(0.1**0.5)*torch.randn(x.shape))
        print(loss.shape)
        loss = loss.mean()
        print(loss)
        grad = torch.autograd.grad(loss, parameters)
        print('g')
        print(grad[-1])
        print(parameters[-1])
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
        return flop*len(x)
    return func


class FedAvg(Server):
    step = 0

    def train(self):
        logger.info("Train with {} workers...".format(self.clients_per_round))
        for r in range(self.num_rounds):
            if r % self.eval_every == 0:
                logger.info("-- Log At Round {} --".format(r))
                if r<= warmup:
                    stats = self.testAE()
                else:
                    stats = self.test()
                if self.eval_train:
                    if r<=warmup:
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

            for idx, c in enumerate(active_clients):
                c.set_param(self.model.get_param())
                if r <= warmup:
                    soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=step_func2)  # stats has (byte w, comp, byte r)
                else:
                    soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=step_func)  # stats has (byte w, comp, byte r)
                soln = [1.0, soln[1]]
                w += soln[0]
                if len(csolns) == 0:
                    csolns = {x: soln[1][x].detach()*soln[0] for x in soln[1]}
                else:
                    for x in csolns:
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
