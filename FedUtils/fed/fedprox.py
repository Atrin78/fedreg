from functools import partial
from .server import Server
from loguru import logger
import numpy as np
from FedUtils.models.utils import decode_stat
import torch


def step_func(model, data, fed):
    lr = model.learning_rate
    parameters = list(model.net.parameters()) + list(model.bottleneck.parameters()) + list(model.head.parameters())
    flop = model.flop
    gamma, old_parameters = fed.gamma, list(fed.model.parameters())

    def func(d):
        nonlocal lr, flop, gamma
        model.train()
        model.zero_grad()
        x, y = d
        pred = model.forward(x)
        loss = model.loss(pred, y).mean()
        for p, op in zip(parameters, old_parameters):
            loss += ((p-op.detach())**2).sum()*gamma
        grad = torch.autograd.grad(loss, parameters)
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
        return flop*len(x)  # only consider the flop in NN
    return func


class FedProx(Server):
    def train(self):
        logger.info("Train with {} workers...".format(self.clients_per_round))
        for r in range(self.num_rounds):
            if r % self.eval_every == 0:
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

                global_stats = self.local_acc(self.model)

            indices, selected_clients = self.select_clients(r, num_clients=self.clients_per_round)
            np.random.seed(r)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round*(1.0-self.drop_percent)), replace=False)

            csolns = {}
            w = 0
            self.global_classifier = self.model.get_classifier()
            self.local_classifier = []
            self.F_in = []
            self.F_out = []
            self.CKA = []



            for idx, c in enumerate(active_clients):
                c.set_param(self.model.get_param())
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=partial(step_func, fed=self))  # stats has (byte w, comp, byte r)
                soln = [1.0, soln[1]]
                w += soln[0]
                if len(csolns) == 0:
                    csolns = {x: soln[1][x].detach()*soln[0] for x in soln[1]}
                else:
                    for x in csolns:
                        csolns[x].data.add_(soln[1][x]*soln[0])
                if r % self.eval_every == 0:
                    self.local_classifier.append(c.model.get_classifier())
                    self.CKA.append(c.get_cka(self.model))
                    local_stats = self.local_acc(c.model)
                    self.local_forgetting(c.id , global_stats, local_stats)
                del c
            
            csolns = [[w, {x: csolns[x]/w for x in csolns}]]

            self.latest_model = self.aggregate(csolns)

            if r % self.eval_every == 0:
                self.compute_divergence()
                self.compute_cka()
                self.compute_forgetting()

            

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
