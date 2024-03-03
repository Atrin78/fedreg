from FedUtils.models.mnist.cnn3 import Model
import torch
from functools import partial
from FedUtils.fed.fedimpress import FedImpress
import torchvision.transforms as transforms
import torchvision


transform_fun = transforms.Compose(
            [
             torchvision.transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: torch.stack([torch.unsqueeze(x, -1),torch.unsqueeze(x, -1),torch.unsqueeze(x, -1)],2))
             ])

config = {

    "seed": 1,
    "model": partial(Model, learning_rate=0.1),
    "inner_opt": None,
    "optimizer": FedImpress,
    "model_param": (2,),
    "inp_size": (128*128*3,),
    "train_path": "retinal",
    "test_path": ["retinal"],
    "clients_per_round": 4,
    "num_rounds": 40,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 5,
    "batch_size": 30,
    "use_fed": 1,
    "log_path": "tasks/retina/FedImpress_e40_lr1/train.log",
    "data_size": 400,

    "train_transform": None,
    "test_transform": None,
    "eval_train": True,



}
