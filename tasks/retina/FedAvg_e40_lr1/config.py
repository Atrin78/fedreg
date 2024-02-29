from FedUtils.models.mnist.cnn import Model
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
    "model": partial(Model, learning_rate=0.005),
    "inner_opt": None,
    "optimizer": FedAvg,
    "model_param": (10,),
    "inp_size": (784,),
    "train_path": "retina",
    "test_path": ["retina"],
    "clients_per_round": 10,
    "num_rounds": 500,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 40,
    "batch_size": 10,
    "use_fed": 1,
    "log_path": "tasks/retina/FedAvg_e40_lr1/train.log",
	"data_size": 400,

    "train_transform": None,
    "test_transform": None,
    "eval_train": True,



}
