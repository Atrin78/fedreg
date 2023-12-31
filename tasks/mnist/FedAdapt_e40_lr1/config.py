from FedUtils.models.mnist.cnn import Model
import torch
from functools import partial
from FedUtils.fed.fedadapt import FedAdapt
import torchvision.transforms as transforms
import torchvision

transform_fun = transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Lambda(lambda x: torch.squeeze(x, -1)),
       #     torchvision.transforms.Resize(32),
       #      torchvision.transforms.RandomRotation((-30, 30)),
       #      transforms.Lambda(lambda x: torch.stack([torch.squeeze(x, 1),torch.squeeze(x, 1),torch.squeeze(x, 1)],1)/3.0)
       #      transforms.Lambda(lambda x: torch.stack([torch.unsqueeze(x, -1),torch.unsqueeze(torch.zeros_like(x), -1),torch.unsqueeze(torch.zeros_like(x), -1)],2))
             ])

config = {

    "seed": 1,
    "model": partial(Model, learning_rate=0.1),
    "inner_opt": None,
    "optimizer": FedAdapt,
    "model_param": (10,),
    "inp_size": (784,),
    "train_path": "data/mnist_10000/data/train/",
    "test_path": ["data/mnist_10000/data/valid/", "data/mnist_10000/data/test/"],
    "clients_per_round": 10,
    "num_rounds": 500,
    "eval_every": 5,
    "drop_percent": 0.0,
    "num_epochs": 40,
    "batch_size": 10,
    "use_fed": 1,
    "log_path": "tasks/mnist/FedAdapt_e40_lr1/train.log",

    "train_transform": None,
    "test_transform": None,
    "eval_train": True,



}
