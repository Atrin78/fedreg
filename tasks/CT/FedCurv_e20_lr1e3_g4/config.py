from FedUtils.models.CT.densenet import DenseNetModel
import torch
from functools import partial
from FedUtils.fed.fedcurv import FedCurv
from torchvision import transforms, utils

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224,), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

# Normalize test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

config = {

    "seed": 1,
    "model": partial(DenseNetModel, learning_rate=1e-3),
    "inner_opt": None,
    "optimizer": FedCurv,
    "model_param": (2,),
    "inp_size": (3*244*244,),
    "train_path": "data/COVID-CT/train/",
    "test_path": ["data/COVID-CT/valid/", "data/COVID-CT/test/"],
    "clients_per_round": 10,
    "num_rounds": 10,
    "eval_every": 1,
    "drop_percent": 0.0,
    "num_epochs": 20,
    "batch_size": 10,
    "use_fed": 1,
    "log_path": "tasks/CT/FedCurv_e20_lr1e3_g4/train.log",
    "train_transform": transform_train,
    "test_transform": transform_test,
    "eval_train": True,
    "gamma": 1e-4


}
