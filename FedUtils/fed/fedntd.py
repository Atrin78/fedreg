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
from .client import Client
from functools import partial
from collections import OrderedDict




def step_func(global_model, model ,data):
    lr = model.learning_rate
    parameters = list(model.net.parameters()) + list(model.bottleneck.parameters()) + list(model.head.parameters())
    logger.info("Parameters: {}".format(len(parameters)))
    flop = model.flop
    global_model = global_model

    def func(d):
        nonlocal flop, lr
        model.train()
        model.zero_grad()
        x, y = d
        y = y.type(torch.LongTensor)
    #    x = torch.reshape(torchvision.transforms.functional.rotate(torch.reshape(x, (-1, 28, 28)), np.random.uniform(-1, 1)), (-1, 784))
        pred = model.forward(x)
        with torch.no_grad():
            global_pred = global_model.forward(x)
            
        # loss = self.criterion(logits, targets, dg_logits)
        loss = model.loss_NTD(pred, y, global_pred)

        grad = torch.autograd.grad(loss, parameters)
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
        return flop*len(x)
    return func




class FedNtd(Server):
    step = 0

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

                global_stats = self.local_acc_loss(self.model)
            indices, selected_clients = self.select_clients(r, num_clients=self.clients_per_round)
            np.random.seed(r)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round*(1.0-self.drop_percent)), replace=False)
            csolns = {}
            w = 0
            self.global_classifier = list(self.model.head.parameters())
            logger.info("Global classifier: {}".format(len(self.global_classifier)))
            self.global_feature_extractor = list(self.model.net.parameters()) + list(self.model.bottleneck.parameters())
            logger.info("Global feature_extracto: {}".format(len(self.global_feature_extractor)))
            logger.info("Global model: {}".format(self.model))
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
                global_model = copy.deepcopy(self.model)
                if torch.cuda.device_count() > 0:
                    global_model.cuda()
                for params in global_model.parameters():
                    params.requires_grad = False
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=partial(step_func, global_model))  # stats has (byte w, comp, byte r)
                soln = [1.0, soln[1]]
                w += soln[0]
                if len(csolns) == 0:
                    csolns = {x: soln[1][x].detach()*soln[0] for x in soln[1]}
                else:
                    for x in csolns:
                        csolns[x].data.add_(soln[1][x]*soln[0])
                if r % self.eval_every == 0:
                    temp = list(c.model.net.parameters()) + list(c.model.bottleneck.parameters())

                    for i,l in enumerate(self.global_feature_extractor):
                        self.local_feature_extractor[i].append(temp[i])  # Append the value to the list for this key
                        
                    temp = list(c.model.head.parameters()) 
                    for i,l in enumerate(self.global_classifier):
                        self.local_classifier[i].append(temp[i])  # Append the value to the list for this key

                    cka_value = c.get_cka(self.model)
                    if cka_value != None:
                        self.CKA.append(c.get_cka(self.model))
                    local_stats = self.local_acc_loss(c.model)
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
