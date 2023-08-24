from FedUtils.models.mnist.cnn2 import Model
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
    "model": partial(Model, learning_rate=1e-1),
    "inner_opt": None,
    "optimizer": FedImpress,
    "model_param": (11,),
    "inp_size": (784,),
    "train_path": "data/mnist_10000/data/train/",
    "test_path": ["data/mnist_10000/data/valid/", "data/mnist_10000/data/test/"],
    "clients_per_round": 10,
    "num_rounds": 100,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 40,
    "batch_size": 10,
    "use_fed": 1,
    "log_path": "tasks/mnist/FedImpress_e40_lr1/train.log",

    "train_transform": transform_fun,
    "test_transform": transform_fun,
    "eval_train": True,



}
