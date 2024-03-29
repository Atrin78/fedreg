from .server import Server
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
#from torch_cka import CKA
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
class_num=2
synthesize_label='cond'
iters_admm= 3
iters_img=30
param_gamma=0.001 
param_admm_rho=0.2
add_bn_normalization = True
lr_img = 100
momentum_img = 0.9
data_size= 50
warmup = 0
reg_coef=4

def step_func(model, data, head):
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
        loss = loss.mean()

        l2_reg = 0
        for p, q in zip(list(model.head.parameters()), head):
            q.requires_grad_(False)
            vec = (p-q)**2
            if len(vec.shape) > 1:
                vec = torch.sum(vec, dim=1)
            l2_reg += torch.mean(vec)

        loss = loss + reg_coef*l2_reg

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

    
def generate_admm(gen_loader, src_model, device, class_num, synthesize_label, iters_admm, iters_img, param_gamma, param_admm_rho, batch_size, add_bn_normalization=False, mode='train'):

    src_model.eval()
    LAMB = torch.zeros_like(src_model.head.weight.data).to(device)
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
        
        for j in range(5):
        #print(gen_dataset.cpu().detach().numpy().shape)
            im = gen_dataset[j][0].cpu().detach().numpy()
            im = (im - np.min(im))/(np.max(im) - np.min(im))
            plt.imsave('imgs/pic'+str(j)+'-'+str(i)+'.jpg', im)

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
                new_matrix = grad_matrix + param_gamma * src_model.head.weight.data
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

        new_matrix = grad_matrix / len(gen_dataset) + param_gamma * src_model.head.weight.data
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
        last_clients = None
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
            csolns = {}
            w = 0

            transform_cifar = transforms.Compose(
            [
           #  torchvision.transforms.functional.rgb_to_grayscale,
             transforms.ToTensor(),
             torchvision.transforms.Resize(128),
             ])
            transform_usps = transforms.Compose(
            [
             torchvision.transforms.Resize(128),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: torch.cat([x,x,x],0)),
             ])

            if r >= warmup:
            #    cifar = torchvision.datasets.CIFAR10(root='./data', train=True,
            #                                      download=True, transform=transform_cifar)
            #    cifar = torch.utils.data.Subset(cifar, list(range(data_size)))
            #    gen_loader = torch.utils.data.DataLoader(cifar, batch_size=self.batch_size, shuffle=True)
                usps = torchvision.datasets.USPS(root='./data', train=True, download=True, transform=transform_usps)
                usps = torch.utils.data.Subset(usps, list(range(data_size)))
                gen_loader = torch.utils.data.DataLoader(usps, batch_size=self.batch_size, shuffle=True)
                gen_dataset, gen_labels, original_dataset ,original_labels = generate_admm(gen_loader, self.model, device, class_num, synthesize_label, iters_admm, iters_img, param_gamma, param_admm_rho, self.batch_size)
                gen_x = gen_dataset
                gen_y = gen_labels
                gen_dataset = torch.tensor(gen_dataset)
                gen_labels = torch.tensor(gen_labels)
                vir_dataset = TensorDataset(gen_dataset, gen_labels)

            vis_x = None
            vis_y = None
            for idx, c in enumerate(active_clients):
                data = iter(DataLoader(c.train_dataset, batch_size=100))
                x,y = next(data)
                if vis_x is None:
                    vis_x = x.cpu().detach().numpy()
                    vis_y = y.cpu().detach().numpy()
                else:
                    vis_x = np.concatenate((vis_x, x.cpu().detach().numpy()), 0)
                    vis_y = np.concatenate((vis_y, y.cpu().detach().numpy()), 0)
            print('dim')
            print(vis_x.shape)
            print(vis_y.shape)
            print(gen_x.shape)
            print(gen_y.shape)
            vis_x = torch.tensor(vis_x)
            _,vis_x = self.model.forward_emb(vis_x)
            _,gen_x = self.model.forward_emb(gen_x)
            ll = list(np.concatenate((vis_y, gen_y.cpu().detach().numpy())))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            ll = [colors[int(inn)] for inn in ll]
            ll_vis = ll[:len(vis_y)]
            ll_gen = ll[len(vis_y):]
            emb = TSNE(n_components=2, perplexity=5).fit_transform(np.concatenate((vis_x.cpu().detach().numpy(), gen_x.cpu().detach().numpy())))
            x,y = emb.T
            x_vis = x[:len(vis_x)]
            x_gen = x[len(vis_x):]
            y_vis = y[:len(vis_x)]
            y_gen = y[len(vis_x):]

          #  sz = [10 if inn<len(vis_x) else 20 for inn in range(len(vis_x)+len(gen_x))]
         #   mrk = ['o' if inn<len(vis_x) else 's' for inn in range(len(vis_x)+len(gen_x))]
            fig, ax = plt.subplots()
            ax.scatter(x_vis,y_vis, marker='o', c=ll_vis, label='original')
            ax.scatter(x_gen,y_gen, marker='s', c=ll_gen, label='generated')
            ax.set_title('round '+str(r))
            ax.legend()
            plt.show()
            plt.savefig('reg2/'+'emb'+str(r)+'.png')
            
           

       #     transform_mnist = transforms.Compose(
       #     [
       #      transforms.ToTensor(),
       #      torchvision.transforms.Resize(32),
       #      transforms.Lambda(lambda x: torch.stack([x,x,x],-1)),
       #      ])
       #     if r >= warmup:
       #         mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
      #          mnist = torch.utils.data.Subset(mnist, list(range(data_size)))
      #          gen_loader = torch.utils.data.DataLoader(mnist, batch_size=self.batch_size*40, shuffle=True)
      #          gen_dataset, gen_labels, original_dataset ,original_labels = generate_admm(gen_loader, self.model, device, class_num, synthesize_label, iters_admm, iters_img, param_gamma, param_admm_rho, self.batch_size)
      #          gen_dataset = torch.tensor(gen_dataset)
      #          gen_labels = torch.tensor(gen_labels)
      #          vir_dataset = TensorDataset(gen_dataset, gen_labels)

            # gen_data = torch.utils.data.DataLoader(vir_dataset, batch_size=self.batch_size, shuffle=True)

            for idx, c in enumerate(active_clients):
                c.set_param(self.model.get_param())
    #            c.set_public()
                #if idx==0:
                #    c.rotate=True
                if r>= warmup:
                    c.gen_data = vir_dataset 
                #glob_dataset = None
                #_, cs = self.select_clients(r+10, num_clients=5)
                #for cl in cs:
                #    if glob_dataset is None:
                #        glob_dataset = cl.train_dataset
                #    else:
                #        glob_dataset = torch.utils.data.ConcatDataset([glob_dataset, cl.train_dataset])
                #c.gen_data = glob_dataset
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, step_func=partial(step_func, head=list(self.model.head.parameters())))  # stats has (byte w, comp, byte r)

                #if last_clients is not None:
                #    print(c.id)
                #    stats_clients = self.local_train_error_and_loss_clients(c.model, last_clients)
                #    logger.info("-- Last Client RESULTS --")
                #    decode_stat(stats_clients)

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
            if last_clients is not None:
                stats_clients = self.train_error_and_loss_clients(last_clients)
                logger.info("-- Last Client RESULTS --")
                decode_stat(stats_clients)
            last_clients = active_clients
            stats_clients = self.train_error_and_loss_clients(active_clients)
            logger.info("-- Active Client RESULTS --")
            decode_stat(stats_clients)

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
