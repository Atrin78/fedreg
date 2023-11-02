from FedUtils.models.mnist.cnn import Model
import torch
from functools import partial
from FedUtils.fed.fedavg_local_eval import FedAvg

config = {

    "seed": 1,
    "model": partial(Model, learning_rate=1e-1),
    "inner_opt": None,
    "optimizer": FedAvg,
    "model_param": (10,),
    "inp_size": (784,),
    "train_path": "/scratch/ssd004/scratch/sayromlou/fedreg_data/data/mnist_10000/data/train/",
    "test_path": ["/scratch/ssd004/scratch/sayromlou/fedreg_data/data/mnist_10000/data/valid/", "/scratch/ssd004/scratch/sayromlou/fedreg_data/data/mnist_10000/data/test/"],
    "clients_per_round": 10,
    "num_rounds": 21,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 10,
    "batch_size": 50,
    "number_of_clients": 100,
    "partition_method": "lda",
    "shard_per_user": 2,
    "alpha": 0.1,
    "root": "/scratch/ssd004/scratch/sayromlou/datasets",
    "dataset_name": "mnist",
    "use_fed": 1,
    "log_path": "tasks_mine/mnist/FedAvg_e10_lr1/train_pretrain_new_data_2.log",
    "save_path": "/scratch/ssd004/scratch/sayromlou/fedreg_models/mnist/FedAvg_e10_lr1_pretrain_new_data_2.pt",
    "load_path":  "/scratch/ssd004/scratch/sayromlou/fedreg_models/mnist/FedAvg_e10_lr1_pretrain_new_data_2.pt_19.pt",

    "train_transform": None,
    "test_transform": None,
    "eval_train": True,



}
