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
import copy


def step_func(model ,data, global_model):
    lr = model.learning_rate
    parameters = list(model.net.parameters()) + list(model.bottleneck.parameters()) + list(model.head.parameters())
    flop = model.flop
    global_model = global_model

    def func(d):
        nonlocal flop, lr
        model.train()
        model.zero_grad()
        x, y = d
    #    x = torch.reshape(torchvision.transforms.functional.rotate(torch.reshape(x, (-1, 28, 28)), np.random.uniform(-1, 1)), (-1, 784))
        pred = model.forward(x)
        with torch.no_grad():
            global_pred = global_model(data)
            
        # loss = self.criterion(logits, targets, dg_logits)
        loss = model.loss_NTD(pred, y, global_pred)

        grad = torch.autograd.grad(loss, parameters)
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
        return flop*len(x)
    return func




class FedNtd(Server):
    step = 0

    def __init__(self, config, Model, datasets, train_transform=None, test_transform=None, traincusdataset=None, evalcusdataset=None, publicdataset=None):
        super(Server, self).__init__(config, Model, datasets, train_transform=train_transform, test_transform=test_transform, traincusdataset=traincusdataset, evalcusdataset=evalcusdataset, publicdataset=publicdataset)
        for c in self.cmodel:
            c.global_model_activate = True

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

            indices, selected_clients = self.select_clients(r, num_clients=self.clients_per_round)
            np.random.seed(r)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round*(1.0-self.drop_percent)), replace=False)
            print('active')
            print([c.id for c in active_clients])
            csolns = {}
            w = 0



            for idx, c in enumerate(active_clients):
                c.set_global_model(copy.deepcopy(self.model))
                c.set_param(self.model.get_param())
                logger.info(f"global model {c.global_model}")
                coef=1
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=step_func, coef=coef)  # stats has (byte w, comp, byte r)
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
