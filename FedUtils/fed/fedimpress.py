from .server_mine import Server
from loguru import logger
import numpy as np
from FedUtils.models.utils import decode_stat
import torch
import gc
from typing import Tuple, Optional, List, Dict
from torch import nn, optim
from torch.optim import SGD
import torch.nn.functional as func
from functools import partial
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import torchvision
import itertools
from FedUtils.models.utils import read_data, CusDataset, ImageDataset
from torch.utils.data import DataLoader
import copy
from collections import OrderedDict
from torch_cka import CKA

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
class_num=10
synthesize_label='cond'
iters_admm=5
iters_img=30
param_gamma=0.001 
param_admm_rho=0.02
add_bn_normalization = True
lr_img = 0.1
momentum_img = 0.9
data_size= 500
warmup = 0

def step_func(model, data):
    lr = model.learning_rate
    parameters = list(model.parameters())
    flop = model.flop

    def func(d, w):
        nonlocal flop, lr
        model.train()
        model.zero_grad()
        x, y = d
        pred = model.forward(x)
        loss = torch.mul(model.loss(pred, y), w)
        #print(w)
        #print(loss.mean())
        loss = loss.mean()
        grad = torch.autograd.grad(loss, parameters)
        total_norm = 0
        for p, g in zip(parameters, grad):
            p.data.add_(-lr*g)
            total_norm += torch.norm(g)**2
    #    print(w)
    #    print(total_norm)
        return flop*len(x)
    return func


def labels_to_one_hot(labels, num_class, device):
    # convert labels to one-hot
    labels_one_hot = torch.FloatTensor(labels.shape[0], num_class).to(device)
    labels_one_hot.zero_()
    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return labels_one_hot

    
def generate_admm(gen_loader, src_model, device, class_num, synthesize_label, iters_admm, iters_img, param_gamma, param_admm_rho, batch_size, add_bn_normalization=True, mode='train'):

    src_model.eval()
    LAMB = torch.zeros_like(src_model.head.linear.weight.data).to(device)
    gen_dataset = None
    gen_labels = None
    original_dataset = None
    original_labels = None
    if add_bn_normalization:
        loss_r_feature_layers = []
        for module in src_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))


    for batch_idx, (images_s, labels_real) in enumerate(gen_loader):
        print(batch_idx,len(images_s))
        images_s = images_s.to(device)
        y_s,_ = src_model.forward_emb(images_s)
        labels_s = y_s.argmax(dim=1)
        if gen_dataset == None:
            gen_dataset = images_s
            if synthesize_label == 'cond':
                gen_labels = torch.tensor(np.random.choice(class_num, len(labels_real)))
            elif synthesize_label == 'pred' or mode == 'test':
                gen_labels = labels_s
            else:
                print('hereee')
                gen_labels = labels_real
            original_dataset = images_s
            original_labels = labels_real
        else:
            gen_dataset = torch.cat((gen_dataset, images_s), 0)
            if synthesize_label == 'cond':
                lab = torch.tensor(np.random.choice(class_num, len(labels_real)))
                gen_labels = torch.cat((gen_labels, lab), 0)
            elif synthesize_label == 'pred' or mode == 'test':
                gen_labels = torch.cat((gen_labels, labels_s), 0)
            else:
                gen_labels = torch.cat((gen_labels, labels_real), 0)
            original_dataset = torch.cat((original_dataset, images_s), 0)
            original_labels = torch.cat((original_labels, labels_real), 0)

#    if args.noise_init:
#        print('here we are rand')
#        gen_dataset = torch.from_numpy(np.random.normal(0, 1, (args.data_size, 3, 256, 256))).type(torch.FloatTensor)
#        gen_labels = torch.randint(low=0, high=2, size=(args.data_size,))
        # print(gen_dataset[0])
        # gen_dataset = torch.from_numpy(np.random.normal(0, 1
        #                                                 , (args.data_size, 3, 256, 256))).type(torch.FloatTensor)
        # gen_labels = torch.randint(low=0, high=2, size=(args.data_size,))
        # print(gen_dataset[0])
        # print(gen_labels[0:20])
#        original_labels = gen_labels.clone()
#        original_dataset = gen_dataset.clone()

    for i in range(iters_admm):
        

        print(f'admm iter: {i}/{iters_admm}')

        # step1: update imgs
        LAMB = LAMB.clone().detach().to(device)
        for batch_idx, (images_s, labels_s) in enumerate(gen_loader):
            # if batch_idx == 10:
            #     break

            gc.collect()

    #        images_s = images_s.to(device)
    #        labels_s = labels_s.to(device)
            images_s = gen_dataset[batch_idx*batch_size:(batch_idx+1)*batch_size].clone().detach().to(device)
            labels_s = gen_labels[batch_idx*batch_size:(batch_idx+1)*batch_size].clone().detach().to(device)

            # convert labels to one-hot
            plabel_onehot = labels_to_one_hot(labels_s, class_num, device)

            # init src img
            images_s.requires_grad_()
            optimizer_s = SGD([images_s], lr_img, momentum=momentum_img)
            first_run = True
            
            for iter_i in range(iters_img):
                y_s, f_s = src_model.forward_emb(images_s)
                loss = func.cross_entropy(y_s, labels_s)
                p_s = func.softmax(y_s, dim=1)
                grad_matrix = (p_s - plabel_onehot).t() @ f_s / p_s.size(0)
                new_matrix = grad_matrix + param_gamma * src_model.head.linear.weight.data
                grad_loss = torch.norm(new_matrix, p='fro') ** 2
                loss += grad_loss * param_admm_rho / 2
                loss += torch.trace(LAMB.t() @ new_matrix)
#                 if args.add_bn_normalization:
#                     rescale = [10] + [1. for _ in range(len(loss_r_feature_layers)-1)]
#                     # if iteration_loc == 0:
#                     #     print("rescale",rescale)
#                     loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

#                     loss += 0.01 * loss_r_feature


      #          if batch_idx==0:
      #              metrics = {"Loss_Genration"+str(i): loss}
      #              wandb.log(metrics)
                first_run = False

                optimizer_s.zero_grad()
                loss.backward()
                optimizer_s.step()

                print(loss)

                # images_s.clamp(0.0, 1.0)
                gc.collect()


            # update src imgs
            gen_dataset[batch_idx*batch_size:(batch_idx+1)*batch_size] = images_s
       #     for img, path in zip(images_s.detach_().cpu(), paths):
       #         torch.save(img.clone(), path)

        # step2: update LAMB
        grad_matrix = torch.zeros_like(LAMB).to(device)
        for batch_idx, (images_s, labels_s) in enumerate(gen_loader):
            # if batch_idx == 10:
            #     break
       #     images_s = images_s.to(device)
       #     labels_s = labels_s.to(device)
            images_s = gen_dataset[batch_idx*batch_size:(batch_idx+1)*batch_size].clone().detach().to(device)
            labels_s = gen_labels[batch_idx*batch_size:(batch_idx+1)*batch_size].clone().detach().to(device)

            # convert labels to one-hot
            plabel_onehot = labels_to_one_hot(labels_s, class_num, device)

            y_s, f_s = src_model.forward_emb(images_s)
            p_s = func.softmax(y_s, dim=1)
            grad_matrix += (p_s - plabel_onehot).t() @ f_s

        new_matrix = grad_matrix / len(gen_dataset) + param_gamma * src_model.head.linear.weight.data
        LAMB += new_matrix * param_admm_rho

        gc.collect()
        
    if add_bn_normalization:
        for hook in loss_r_feature_layers:
            hook.close()


 #   if (a_iter-1) % args.save_every == 0:
 #       print("saving image dir to", save_dir)
 #       vutils.save_image(torch.cat((original_dataset[0:20],gen_dataset[0:20]),0), save_dir ,
  #                        normalize=True, scale_each=True, nrow=int(10))
        # plt.style.use('dark_background')
        # # fig = plt.figure()
        # # ax = fig.add_subplot()
        # image = plt.imread(save_dir)
        # ax.imshow(image)
        # ax.axis('off')
        # fig.set_size_inches(10 * 5, 10*10 )
        # plt.title("ori_labels= "+str(original_labels[0:20])+"\n gen_labels="+str(gen_labels[0:20]), fontweight="bold")
        # plt.savefig(save_dir)
        # plt.close()

    return gen_dataset, gen_labels, original_dataset ,original_labels




class FedImpress(Server):
    step = 0

    def train(self):

        logger.info("Train with {} workers...".format(self.clients_per_round))
        for r in range(self.start_round+1,self.num_rounds):
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

            transform_mnist = transforms.Compose(
            [
             transforms.ToTensor(),
             torchvision.transforms.Resize(32),
             transforms.Lambda(lambda x: torch.stack([x,x,x],-1)),
             ])
            if r >= warmup:
                mnist = torchvision.datasets.MNIST(root='./data', train=True,
                                                  download=True, transform=transform_mnist)
                mnist = torch.utils.data.Subset(mnist, list(range(data_size)))
                gen_loader = torch.utils.data.DataLoader(mnist, batch_size=self.batch_size, shuffle=True)
                gen_dataset, gen_labels, original_dataset ,original_labels = generate_admm(gen_loader, self.model, device, class_num, synthesize_label, iters_admm, iters_img, param_gamma, param_admm_rho, self.batch_size)
                gen_dataset = torch.tensor(gen_dataset)
                gen_labels = torch.tensor(gen_labels)
                vir_dataset = TensorDataset(gen_dataset, gen_labels)

            self.model = self.model_type(*self.model_param, self.inner_opt)

            indices, selected_clients = self.select_clients(r, num_clients=self.clients_per_round)
            np.random.seed(r)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round*(1.0-self.drop_percent)), replace=False)
            csolns = {}
            list_clients = {}
            w = 0
            self.global_classifier = list(self.model.head.linear.parameters())
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
                if r>= warmup:
                    c.gen_data = vir_dataset
                coef=1
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=step_func)  # stats has (byte w, comp, byte r)
                soln = [1.0, soln[1]]
                w += soln[0]
                local_stats = self.local_acc_loss(c.model)
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
                    #local_stats = self.local_acc_loss(c.model)
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

# for p in model.net.parameters():
#         p.requires_grad = False
#     if model.bottleneck != None:
#         for p in model.bottleneck.parameters():
#             p.requires_grad = False
#     for p in model.head.parameters():
#         p.requires_grad = True