from FedUtils.models.mnist.cnn3 import Model
from functools import partial
from FedUtils.fed.fedreg import FedReg
from torch.optim import SGD
import torchvision.transforms as transforms
import torchvision
import torch


transform_fun = transforms.Compose(
            [
             torchvision.transforms.Resize((28, 28)),
             transforms.ToTensor(),
             #transforms.Lambda(lambda x: torch.stack([torch.unsqueeze(x, -1),torch.unsqueeze(x, -1),torch.unsqueeze(x, -1)],2)/3.0)
             ])

config = {

    "seed": 1,
    "model": partial(Model, learning_rate=1e-4, p_iters=10, ps_eta=2e-1, pt_eta=2e-3),
    "inner_opt": None,
    "optimizer": FedReg,
    "model_param": (2,),
    "inp_size": (128*128*3,),
    "train_path": "retina",
    "test_path": ["retina"],
    "clients_per_round": 3,
    "num_rounds": 80,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 5,
    "batch_size": 30,
    "use_fed": 1,
    "log_path": "tasks/retina/FedReg_e40_lr1_g4/train.log",
    "train_transform": None,
    "test_transform": None,
    "eval_train": True,
    "data_size":100,
    "gamma": 0.4,

}
