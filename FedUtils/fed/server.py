from torch import nn
import numpy as np
from .client import Client
import torch
import copy
from loguru import logger
from FedUtils.models.utils import decode_stat

class Server(object):
    def __init__(self, config, Model, datasets, train_transform=None, test_transform=None, traincusdataset=None, evalcusdataset=None, publicdataset=None):
        super(Server, self).__init__()
        self.config = config
        self.model_param = config["model_param"]
        self.inner_opt = config["inner_opt"]
        self.clients_per_round = config["clients_per_round"]
        self.num_rounds = config["num_rounds"]
        self.eval_every = config["eval_every"]
        self.batch_size = config["batch_size"]
        self.drop_percent = config["drop_percent"]
        self.num_epochs = config["num_epochs"]
        self.eval_train = config["eval_train"]
        if "gamma" in config:
            self.gamma = config["gamma"]
        else:
            self.gamma = 1.0

        if "add_mask" in config:
            self.add_mask = config["add_mask"]
        else:
            self.add_mask = -1
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.model = Model(*self.model_param, self.inner_opt)
        self.cmodel = Model(*self.model_param, self.inner_opt)
        self.traincusdataset = traincusdataset
        self.evalcusdataset = evalcusdataset
        self.clients = self.__set_clients(datasets, Model)
        self.F_in = []
        self.F_out = []
        self.classifier_epsilpon = 0
        self.CKA = 0
        self.publicdataset = publicdataset
        self.global_classifier = []
        self.local_classifier = []
        self.maybe_active_clients_global_model()

    def maybe_active_clients_global_model(self):
        return

    def __set_clients(self, dataset, Model):
        users, groups, train_data, test_data = dataset
        logger.info("Number of clients: {}".format(len(train_data)))
        logger.info("Number of clients: {}".format(len(test_data[0])))
        for i in range(len(test_data)):
            logger.info('hereeee')
            logger.info("Number of clients: {}".format(test_data[i]))
            logger.info("Number of clients: {}".format(train_data))
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [(u, g, train_data[u], [td[u] for td in test_data], Model, self.batch_size, self.train_transform, self.test_transform) for u, g in zip(users, groups)]
        for i in all_clients:
            logger.info("check setting: {}".format(len(i[3][0]['x'])))
        return all_clients

    def set_param(self, state_dict):
        self.model.set_param(state_dict)
        return True

    def get_param(self):
        return self.model.get_param()

    def _aggregate(self, wstate_dicts):
        old_params = self.get_param()
        state_dict = {x: 0.0 for x in self.get_param() if x.split('.')[0]!='adapt'}
        wtotal = 0.0
        for w, st in wstate_dicts:
            wtotal += w
            for name in state_dict.keys():
                if name.split('.')[0]=='adapt':
                    continue
                assert name in state_dict
                state_dict[name] += st[name]*w
        state_dict = {x: state_dict[x]/wtotal for x in state_dict}
        return state_dict

    def aggregate(self, wstate_dicts):
        old = self.model.get_param()
        state_dict = self._aggregate(wstate_dicts)
        total=0
        for key in state_dict:
            diff = old[key].detach().cpu()-state_dict[key].detach().cpu()
            total += torch.norm(diff)**2
        print(total)
        return self.set_param(state_dict)

    def select_clients(self, seed, num_clients=20):
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(seed)
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        clients = [self.clients[c] for c in indices]
        clients = [Client(c[0], c[1], c[2], c[3], self.cmodel, c[5], c[6], c[7], self.traincusdataset, self.evalcusdataset) for c in clients]
        return indices, clients

    def save(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        num_samples = []
        tot_correct = []
        for i in self.clients:
            logger.info("check test: {}".format(len(i[3][0]['x'])))
        clients = [x for x in self.clients if len(x[3][0]['x']) > 0]
        logger.info("test clients: {}".format(len(clients)))
        clients = [Client(c[0], c[1], c[2], c[3], self.cmodel, c[5], c[6], c[7], self.traincusdataset, self.evalcusdataset) for c in clients]
        for m in clients:
            m.set_param(self.get_param())

        for c in clients:
            ct, ns = c.test()
            tot_correct.append(ct)
            num_samples.append(ns)
        ids = [c.id for c in clients]
        groups = [c.group for c in clients]
        num_test = len(tot_correct[0])
        tot_correct = [[a[i] for a in tot_correct] for i in range(num_test)]
        num_samples = [[a[i] for a in num_samples] for i in range(num_test)]
        return ids, groups, num_samples, tot_correct
    
    def local_acc(self,model):
        num_samples = []
        tot_correct = []
        clients = [x for x in self.clients if len(x[3][0]['x']) > -1]
        logger.info("clients: {}".format(len(clients)))
        clients = [Client(c[0], c[1], c[2], c[3], self.cmodel, c[5], c[6], c[7], self.traincusdataset, self.evalcusdataset) for c in clients]

        for m in clients:
            m.set_param(model.get_param())

        for c in clients:
            ct, ns = c.test()
            tot_correct.append(ct)
            num_samples.append(ns)


        ids = [c.id for c in clients]
        logger.info("ids: {}".format(ids))
        groups = [c.group for c in clients]
        logger.info("tot_correct: {}".format(tot_correct))
        num_test = len(tot_correct[0])
        tot_correct = [[a[i] for a in tot_correct] for i in range(num_test)]
        num_samples = [[a[i] for a in num_samples] for i in range(num_test)]
        return ids, groups, num_samples, tot_correct

    
    def local_forgetting(self, client_id, stats, local_stats):
        local_ids, local_groups, local_num_samples, local_tot_correct = local_stats
        global_ids, global_groups, global_num_samples, global_tot_correct = stats
        logger.info("local_ids: {}".format(local_ids))
        if len(stats) == 4:
            local_ids, local_groups, local_num_samples, local_tot_correct = local_stats
            global_ids, global_groups, global_num_samples, global_tot_correct = stats
        elif len(stats) == 5:
            local_ids, local_groups, local_num_samples, local_tot_correct, local_loss = local_stats
            global_ids, global_groups, global_num_samples, global_tot_correct, gloal_loss = stats
        else:
            raise ValueError
        if isinstance(local_num_samples[0], list):
            assert len(local_num_samples) == len(local_tot_correct)
            F_in_list = []
            F_out_list = []
            for i in range(len(local_tot_correct)):
                F_in = 0
                F_out = 0
                num_out = 0
                for j in range(len(local_tot_correct[i])):
                    if local_ids[j] == client_id:
                        F_in +=  (global_tot_correct[i][j] * 1.0 / global_num_samples[i][j]) - (local_tot_correct[i][j] * 1.0 / local_num_samples[i][j])
                    else:
                        F_out +=  (global_tot_correct[i][j] * 1.0 / global_num_samples[i][j]) - (local_tot_correct[i][j] * 1.0 / local_num_samples[i][j])
                        num_out += 1
            
                F_in_list.append(F_in)
                F_out_list.append(F_out)
            self.F_in.append(F_in_list)
            self.F_out.append(F_out_list)
        else:
            F_in = 0
            F_out = 0
            num_out = 0
            for j in range(len(local_tot_correct)):
                if local_ids[j] == client_id:
                    F_in +=  (global_tot_correct[j] * 1.0 / global_num_samples[j]) - (local_tot_correct[j] * 1.0 / local_num_samples[j])
                else:
                    F_out +=  (global_tot_correct[j] * 1.0 / global_num_samples[j]) - (local_tot_correct[j] * 1.0 / local_num_samples[j])
                    num_out += 1
            self.F_in.append(F_in)
            self.F_out.append(F_out)
        return
    
    def compute_forgetting(self):
        logger.info("Forgetting: {} {}".format(self.F_out, self.F_in))
        if self.F_out[0] is list:
            self.F_out = torch.tensor(self.F_out).reshape(2,-1)
            self.F_in = torch.tensor(self.F_out).reshape(2,-1)
            logger.info("Forgetting: {} {}".format(self.F_out, self.F_in))
            for i in range(len(self.F_out)):
                logger.info("Test_{} Out_forgetting: {}".format(i,torch.sum(self.F_out[i]) / len(self.F_out[i])))
                logger.info("Test_{} In_forgetting: {}".format(i,torch.sum(self.F_in[i]) / len(self.F_in[i])))
        else:
            self.F_out = torch.tensor(self.F_out)
            self.F_in = torch.tensor(self.F_out)
            logger.info("Out_forgetting: {}".format(torch.sum(self.F_out) / len(self.F_out)))
            logger.info("In_forgetting: {}".format(torch.sum(self.F_in) / len(self.F_in)))
        return
    
    def compute_divergence(self):
        up = 0
        down = 0
        for l in self.local_classifier:
            up += torch.sum(l-self.global_classifier)**2
            down += l-self.global_classifier
        divergence = up/torch.sum(down)**2
        logger.info("divergence: {}".format(divergence))
        return
    
    def compute_cka(self):
        if self.CKA[0] is list:
            for i in range(len(self.CKA)):
                logger.info("Test_{} cka: {}".format(i,torch.mean( torch.tensor(self.CKA[i]))))
        else:
            logger.info("cka: {}".format(torch.mean( torch.tensor(self.CKA))))
        return  

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        clients = self.clients
        clients = [Client(c[0], c[1], c[2], c[3], self.cmodel, c[5], c[6], c[7], self.traincusdataset, self.evalcusdataset) for c in clients]
        for m in clients:
            m.set_param(self.get_param())
        for c in clients:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        ids = [c.id for c in clients]
        groups = [c.group for c in clients]
        return ids, groups, num_samples, tot_correct, losses


    def test_adapt(self, step_fun, num_epochs):
        num_samples = []
        tot_correct = []
        clients = [x for x in self.clients if len(x[3][0]['x']) > 0]
        clients = [Client(c[0], c[1], c[2], c[3], self.cmodel, c[5], c[6], c[7], self.traincusdataset, self.evalcusdataset) for c in clients]
        for m in clients:
            m.set_param(self.get_param())
        [m.solve_inner(num_epochs=num_epochs, step_func=step_fun, coef=1) for m in clients]

        for c in clients:
            ct, ns = c.test()
            tot_correct.append(ct)
            num_samples.append(ns)
        ids = [c.id for c in clients]
        groups = [c.group for c in clients]
        num_test = len(tot_correct[0])
        tot_correct = [[a[i] for a in tot_correct] for i in range(num_test)]
        num_samples = [[a[i] for a in num_samples] for i in range(num_test)]
        return ids, groups, num_samples, tot_correct

    def train_error_and_loss_adapt(self, step_fun, num_epochs):
        num_samples = []
        tot_correct = []
        losses = []
        clients = self.clients
        clients = [Client(c[0], c[1], c[2], c[3], self.cmodel, c[5], c[6], c[7], self.traincusdataset, self.evalcusdataset) for c in clients]
        for m in clients:
            m.set_param(self.get_param())
        [m.solve_inner(num_epochs=num_epochs, step_func=step_fun, coef=1) for m in clients]
        for c in clients:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        ids = [c.id for c in clients]
        groups = [c.group for c in clients]
        return ids, groups, num_samples, tot_correct, losses




    def train_error_and_loss_clients(self, clients):
        num_samples = []
        tot_correct = []
        losses = []
        for m in clients:
            m.set_param(self.get_param())
        for c in clients:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        ids = [c.id for c in clients]
        groups = [c.group for c in clients]
        return ids, groups, num_samples, tot_correct, losses

    def local_train_error_and_loss_clients(self, model, clients):
        num_samples = []
        tot_correct = []
        losses = []
        for c in clients:
            ct, cl, ns = c.train_error_and_loss_model(model)
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        ids = [c.id for c in clients]
        groups = [c.group for c in clients]
        return ids, groups, num_samples, tot_correct, losses


    def testAE(self):
        num_samples = []
        tot_correct = []
        clients = [x for x in self.clients if len(x[3][0]['x']) > 0]
        clients = [Client(c[0], c[1], c[2], c[3], self.cmodel, c[5], c[6], c[7], self.traincusdataset, self.evalcusdataset) for c in clients]
        for m in clients:
            m.set_param(self.get_param())

        for c in clients:
            ct, ns = c.testAE()
            tot_correct.append(ct)
            num_samples.append(ns)
        ids = [c.id for c in clients]
        groups = [c.group for c in clients]
        num_test = len(tot_correct[0])
        tot_correct = [[a[i] for a in tot_correct] for i in range(num_test)]
        num_samples = [[a[i] for a in num_samples] for i in range(num_test)]
        return ids, groups, num_samples, tot_correct

    def train_error_and_lossAE(self):
        num_samples = []
        tot_correct = []
        losses = []
        clients = self.clients
        clients = [Client(c[0], c[1], c[2], c[3], self.cmodel, c[5], c[6], c[7], self.traincusdataset, self.evalcusdataset) for c in clients]
        for m in clients:
            m.set_param(self.get_param())
        for c in clients:
            ct, cl, ns = c.train_error_and_lossAE()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        ids = [c.id for c in clients]
        groups = [c.group for c in clients]
        return ids, groups, num_samples, tot_correct, losses
