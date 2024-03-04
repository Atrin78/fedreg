from FedUtils.models.mnist.cnn3 import Model
import torch
from functools import partial
from FedUtils.fed.fedprox import FedProx

config = {

    "seed": 1,
    "model": partial(Model, learning_rate=1e-2),
    "inner_opt": None,
    "optimizer": FedProx,
    "model_param": (2,),
    "inp_size": (128*128*3,),
    "train_path": "retina",
    "test_path": ["retina"],
    "clients_per_round": 4,
    "num_rounds": 40,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 10,
    "batch_size": 60,
    "use_fed": 1,
    "data_size":50,
    "log_path": "tasks/retina/FedProx_e40_lr1_g001/train.log",
    "gamma": 0.001,

    "train_transform": None,
    "test_transform": None,
    "eval_train": True


}
