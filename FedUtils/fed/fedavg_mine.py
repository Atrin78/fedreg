from .server_mine import Server
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
from collections import OrderedDict


def step_func(model, data):
    lr = model.learning_rate
    if model.bottleneck != None:
        parameters = list(model.net.parameters()) + list(model.bottleneck.parameters()) + list(model.head.parameters())
    else:
        parameters = list(model.net.parameters()) + list(model.head.parameters())
    flop = model.flop

    def func(d):
        nonlocal flop, lr
        model.train()
        model.zero_grad()
        x, y = d
    #    x = torch.reshape(torchvision.transforms.functional.rotate(torch.reshape(x, (-1, 28, 28)), np.random.uniform(-1, 1)), (-1, 784))
        pred = model.forward(x)
        loss = model.loss(pred, y).mean()

        grad = torch.autograd.grad(loss, parameters)
        for p, g in zip(parameters, grad):
            if p.requires_grad:
                p.data.add_(-lr*g)
        return flop*len(x)
    return func




class FedAvg(Server):
    step = 0

    def train(self):

        logger.info("Train with {} workers...".format(self.clients_per_round))
        for r in range(self.start_round,self.num_rounds):
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
                self.save_model(r)

                global_stats = self.local_acc_loss(self.model)

            indices, selected_clients = self.select_clients(r, num_clients=self.clients_per_round)
            np.random.seed(r)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round*(1.0-self.drop_percent)), replace=False)
            csolns = {}
            w = 0
            self.global_classifier = list(self.model.head.parameters())
            if self.model.bottleneck != None:
                self.global_feature_extractor = list(self.model.net.parameters()) + list(self.model.bottleneck.parameters())
            else:
                self.global_feature_extractor = list(self.model.net.parameters())
            self.local_classifier = [[] for l in self.global_classifier]
            self.local_feature_extractor = [[] for l in self.global_feature_extractor]
            self.F_in = []
            self.F_out = []
            self.loss_in = []
            self.loss_out = []
            self.CKA = []



            for idx, c in enumerate(active_clients):
                c.set_param(self.model.get_param())
                coef=1
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=step_func)  # stats has (byte w, comp, byte r)
                soln = [1.0, soln[1]]
                w += soln[0]
                if len(csolns) == 0:
                    csolns = {x: soln[1][x].detach()*soln[0] for x in soln[1]}
                else:
                    for x in csolns:
                        csolns[x].data.add_(soln[1][x]*soln[0])
                if r % self.eval_every == 0:
                    pass
                    # if c.model.bottleneck != None:
                    #     temp = list(c.model.net.parameters()) + list(c.model.bottleneck.parameters())
                    # else:
                    #     temp = list(c.model.net.parameters())

                    # for i,l in enumerate(self.global_feature_extractor):
                    #     self.local_feature_extractor[i].append(temp[i])  # Append the value to the list for this key
                        
                    # temp = list(c.model.head.parameters()) 
                    # for i,l in enumerate(self.global_classifier):
                    #     self.local_classifier[i].append(temp[i])  # Append the value to the list for this key

                    # cka_value = c.get_cka(self.model)
                    # if cka_value != None:
                    #     self.CKA.append(c.get_cka(self.model))
                    # local_stats = self.local_acc_loss(c.model)
                    # self.local_forgetting(c.id , global_stats, local_stats)
                del c
            
            csolns = [[w, {x: csolns[x]/w for x in csolns}]]
            self.latest_model = self.aggregate(csolns)

            if r % self.eval_every == 0:
                pass
                # self.compute_divergence()
                # self.compute_cka()
                # self.compute_forgetting()

            

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
