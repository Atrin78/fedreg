from FedUtils.models.transformer.model import Model
import torch
from functools import partial
from FedUtils.fed.fedreg import FedReg
from torchvision import transforms, utils

transform_train = transforms.Compose([
    transforms.Resize(512),
    transforms.RandomCrop(384),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

# Normalize test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.Resize(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])


config = {

    "seed": 1,
    "model": partial(Model, learning_rate=1e-1, p_iters=10, ps_eta=1e-2, pt_eta=1e-4),
    "inner_opt": None,
    "optimizer": FedReg,
    "model_param": (100,),
    "inp_size": (3*32*32,),
    "train_path": "data/cifar-100-python/data/train/",
    "test_path": ["data/cifar-100-python/data/valid/", "data/cifar-100-python/data/test/"],
    "clients_per_round": 500,
    "num_rounds": 100,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 5,
    "batch_size": 5,
    "use_fed": 1,
    "log_path": "tasks/cifar100_transformer/FedReg_e5_lr10/train.log",
    "train_transform": transform_train,
    "test_transform": transform_test,
    "eval_train": False,
    "gamma": 0.02,  # gamma_func,



}
