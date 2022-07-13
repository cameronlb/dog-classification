import os
import random
import torchvision.models
import wandb
import torch

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from tqdm import tqdm
from torch import optim
from torch.utils.data import Dataset, DataLoader
from data_loader import StanfordDogsDataset as StanfordDogs
from data_loader.data_loaders import get_data, make_data_loaders
from test import test
from train import train

DATA_DIR = r"C:\Users\Cameron\Documents\python projects\dog classification\data\stanford_dataset\images"

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using {DEVICE} device")

# Weights and bias login
wandb.login()

config = dict(
    image_size=(32, 32),
    data_split_ratio=0.7,
    epochs=300,
    classes=120,
    batch_size=64,
    learning_rate=0.0001,
    optimizer="Adam",
    loss_function="CrossEntropyLoss",
    dataset="STANFORD_DOGS",
    architecture="CNN",
    pretrained_name = "resnet18")

print(config)


def model_pipeline(hyperparameters):
    with wandb.init(project="pytorch-dog-breed-classifier", entity="cambino", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # set run name
        wandb.run.name = str(config)
        wandb.run.save()

        # make the model, data, and optimization problem
        model, train_dataloader, test_dataloader, loss_fn, optimizer = make(config)
        print(model)

        # and use them to train the model
        train(model, train_dataloader, loss_fn, optimizer, config)

        # and test its final performance
        test(model, test_dataloader)
    return model

def make(config):
    # Make the train and test data
    train_data, test_data = get_data(DATA_DIR, config.image_size, config.data_split_ratio)

    # Make data_loader for train and test
    train_dataloader, test_dataloader = make_data_loaders(train_data, test_data, config.batch_size)

    # Pretrained model
    model = torchvision.models.resnet18(pretrained=True)

    # Put model on CPU or GPU
    model.to(DEVICE)

    # Change pretrained final layer model
    model.fc = nn.Linear(512, config.classes)

    # Make the loss function (criterion) and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    return model, train_dataloader, test_dataloader, loss_fn, optimizer

# Build, train and analyze the model with the pipeline
pet_breed_model = model_pipeline(config)