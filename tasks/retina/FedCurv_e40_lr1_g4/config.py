from FedUtils.models.mnist.cnn3 import Model
import torch
from functools import partial
from FedUtils.fed.fedcurv import FedCurv

config = {

    "seed": 1,
    "model": partial(Model, learning_rate=1e-2),
    "inner_opt": None,
    "optimizer": FedCurv,
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
    "log_path": "tasks/retina/FedCurv_e40_lr1_g4/train.log",
    "data_size":100,
    "gamma": 1e-4,

    "train_transform": None,
    "test_transform": None,
    "eval_train": True


}
