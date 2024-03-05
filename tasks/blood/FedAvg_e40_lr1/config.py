from FedUtils.models.mnist.cnn3 import Model
import torch
from functools import partial
from FedUtils.fed.fedavg import FedAvg
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
    "model": partial(Model, learning_rate=0.01),
    "inner_opt": None,
    "optimizer": FedAvg,
    "model_param": (8,),
    "inp_size": (128*128*3,),
    "train_path": "bloodmnist",
    "test_path": ["bloodmnist"],
    "clients_per_round": 8,
    "num_rounds": 40,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 10,
    "batch_size": 20,
    "use_fed": 1,
    "log_path": "tasks/blood/FedAvg_e40_lr1/train.log",
    "data_size": 400,

    "train_transform": None,
    "test_transform": None,
    "eval_train": True,



}
