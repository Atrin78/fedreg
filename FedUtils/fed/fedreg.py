from .server_mine import Server
from loguru import logger
import numpy as np
from FedUtils.models.utils import decode_stat
import torch
from functools import partial
import copy


def step_func(model, data, fed):
    lr = model.learning_rate
    if model.bottleneck != None:
        parameters = list(model.net.parameters()) + list(model.bottleneck.parameters()) + list(model.head.parameters())
    else:
        parameters = list(model.net.parameters()) + list(model.head.parameters())


    flop = model.flop
    gamma = fed.gamma
    add_mask = fed.add_mask
    beta = 0.5

    psuedo_data, perturb_data = [], []
    for d in data[0]:
        x, y = d
        psuedo, perturb = fed.model.generate_fake(x, y)
        psuedo_data.append(psuedo)
        perturb_data.append(perturb)
    idx = 0
    median_model, old_model, penal_model = copy.deepcopy(fed.model), copy.deepcopy(fed.model), copy.deepcopy(fed.model)
    median_parameters = list(median_model.net.parameters()) + list(median_model.bottleneck.parameters()) + list(median_model.head.parameters())
    old_parameters = list(old_model.net.parameters()) + list(old_model.bottleneck.parameters()) + list(old_model.head.parameters())
    penal_parameters = list(penal_model.net.parameters()) + list(penal_model.bottleneck.parameters()) + list(penal_model.head.parameters())

    def func(d):
        nonlocal idx, add_mask, beta, flop, gamma, lr
        model.train()
        median_model.train()
        penal_model.train()
        model.zero_grad()
        median_model.zero_grad()
        penal_model.zero_grad()
        x, y = d
        psd, ptd = psuedo_data[idx % len(psuedo_data)], perturb_data[idx % len(perturb_data)]
        idx += 1

        for p, m, o in zip(parameters, median_parameters, old_parameters):
            m.data.copy_(gamma*p+(1-gamma)*o)

        mloss = median_model.loss(median_model(x), y).mean()
        grad1 = torch.autograd.grad(mloss, median_parameters)

        if add_mask > 0:
            fnx, fny, pred_fny = old_model.generate_fake(x, y)[0]
            avg_fny = (1.0-0*pred_fny)/pred_fny.shape[-1]
            mask_grad = torch.autograd.grad(median_model.loss(median_model(fnx), avg_fny).mean(), median_parameters)

            sm = sum([(gm * gm).sum() for gm in mask_grad])
            sw = (sum([(g1 * gm).sum() for g1, gm in zip(grad1, mask_grad)])) / sm.add(1e-30)
            grad1 = [a-sw*b for a, b in zip(grad1, mask_grad)]

        for g1, p in zip(grad1, parameters):
            p.data.add_(-lr*g1)

        for p, o, pp in zip(parameters, old_parameters, penal_parameters):
            pp.data.copy_(p*beta+o*(1-beta))

        ploss = penal_model.loss(penal_model(psd[0]), psd[2]).mean()
        grad2 = torch.autograd.grad(ploss, penal_parameters)
        with torch.no_grad():
            dtheta = [(p-o) for p, o in zip(parameters, old_parameters)]
            s2 = sum([(g2*g2).sum() for g2 in grad2])
            w = (sum([(g0*g2).sum() for g0, g2 in zip(dtheta, grad2)]))/s2.add(1e-30)
            w = w.clamp(0.0, )

        pertub_ploss = penal_model.loss(penal_model(ptd[0]), ptd[1]).mean()
        grad3 = torch.autograd.grad(pertub_ploss, penal_parameters)
        s3 = sum([(g3*g3).sum() for g3 in grad3])
        w1 = (sum([((g0-w*g2)*g3).sum() for g0, g2, g3 in zip(dtheta, grad2, grad3)]))/s3.add(1e-30)
        w1 = w1.clamp(0.0,)

        for g2, g3, p in zip(grad2, grad3, parameters):
            p.data.add_(-w*g2-w1*g3)
        if add_mask:
            return flop*len(x)*4  # only consider the flop in NN
        else:
            return flop*len(x)*3
    return func


class FedReg(Server):
    def train(self):
        logger.info("Train with {} workers...".format(self.clients_per_round))
        epochs = self.num_epochs
        last_clients = None
        for r in range(self.start_round+1,self.num_rounds):
            self.round = r


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
            list_clients = {}
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
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=partial(step_func, fed=self))
                soln = [1.0, soln[1]]
                w += soln[0]
                if len(csolns) == 0:
                    for x in soln[1]:
                        csolns[x] = soln[1][x].detach()*soln[0]
                        list_clients[x] = [soln[1][x].detach()*soln[0]]
                else:
                    for x in csolns:
                        csolns[x].data.add_(soln[1][x]*soln[0])
                        list_clients[x].append(soln[1][x].detach()*soln[0])
                if r % self.eval_every == 0:
                    # cka = c.get_cka(self.model)
                    # if cka != None:
                    #     self.CKA.append(cka)
                    local_stats = self.local_acc_loss(c.model)
                    self.local_forgetting(c.id , global_stats, local_stats)
                del c

            if r % self.eval_every == 0:
                # pass
                # self.compute_cka()
                self.compute_forgetting()

            csolns = [[w, {x: csolns[x]/w for x in csolns}]]
            self.compute_divergence(list_clients)
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
