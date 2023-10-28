from FedUtils.models.mnist.cnn import Model
import torch
from functools import partial
from FedUtils.fed.fedprox import FedProx

config = {

    "seed": 1,
    "model": partial(Model, learning_rate=1e-1),
    "inner_opt": None,
    "optimizer": FedProx,
    "model_param": (10,),
    "inp_size": (784,),
    "train_path": "/scratch/ssd004/scratch/sayromlou/fedreg_data/data/mnist_10000/data/train/",
    "test_path": ["/scratch/ssd004/scratch/sayromlou/fedreg_data/data/mnist_10000/data/valid/", "/scratch/ssd004/scratch/sayromlou/fedreg_data/data/mnist_10000/data/test/"],
    "clients_per_round": 10,
    "num_rounds": 500,
    "eval_every": 2,
    "drop_percent": 0.0,
    "num_epochs": 10,
    "batch_size": 10,
    "use_fed": 1,
    "log_path": "tasks_mine/mnist/FedProx_e10_lr1/train.log",
    "gamma": 0.001,

    "train_transform": None,
    "test_transform": None,
    "eval_train": True,



}