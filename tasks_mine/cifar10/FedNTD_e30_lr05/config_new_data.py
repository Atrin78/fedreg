from FedUtils.models.cifar10.resnet9_mine import Model
import torch
from functools import partial
from FedUtils.fed.fedntd import FedNtd
from torchvision import transforms, utils

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Normalize test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

config = {

    "seed": 1,
    "model": partial(Model, learning_rate=5e-3),
    "inner_opt": None,
    "optimizer": FedNtd,
    "model_param": (10,),
    "inp_size": (3*32*32,),
    "train_path": "/scratch/ssd004/scratch/sayromlou/fedreg_data/data/cifar-10-batches-py/data_uni/train/",
    "test_path": ["/scratch/ssd004/scratch/sayromlou/fedreg_data/data/cifar-10-batches-py/data_uni/valid/", "/scratch/ssd004/scratch/sayromlou/fedreg_data/data/cifar-10-batches-py/data_uni/test/"],
    "clients_per_round": 2,
    "num_rounds": 1,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 1,
    "batch_size": 50,
    "number_of_clients": 10,
    "partition_method": "lda",
    "shard_per_user": 2,
    "alpha": 0.1,
    "root": "/scratch/ssd004/scratch/sayromlou/datasets",
    "dataset_name": "cifar10",
    "use_fed": 1,
    "log_path": "tasks_mine/cifar10/FedNTD_e30_lr05/train_pretrain_new_data_final.log",
    "save_path": "/scratch/ssd004/scratch/sayromlou/fedreg_models/cifar10/FedNTD_e30_lr05_pretrain_new_data_final",
    "load_path": None,

    "train_transform": transform_train,
    "test_transform": transform_test,
    "eval_train": True,



}