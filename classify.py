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
from dataloaders import StanfordDogsDataset as StanfordDogs


DATA_DIR = r"C:\Users\Cameron\Documents\python projects\dog classification\data\stanford_dataset\images"

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Weights and bias login
wandb.login()

config = dict(
    image_size=(64, 64),
    data_split_ratio=0.7,
    epochs=3,
    classes=120,
    batch_size=32,
    learning_rate=0.005,
    dataset="STANFORD_DOGS",
    architecture="CNN",
    run_name = "pretrained-resnet18-stanford-dogs")

def model_pipeline(hyperparameters):
    with wandb.init(project="pytorch-dog-breed-classifier", entity="cambino", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # set run name
        wandb.run.name = config.run_name
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

    # Make dataloaders for train and test
    train_dataloader, test_dataloader = make_data_loaders(train_data, test_data, config.batch_size)

    # Make the model
    # model = CustomModel.OxfordPetsModel(config.feature_maps, config.classes).to(device)

    # Pretrained model
    model = torchvision.models.resnet18(pretrained=True)
    model.train(True)

    # Make the loss function (criterion) and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    return model, train_dataloader, test_dataloader, loss_fn, optimizer

def get_data(data_dir, image_size, split_ratio):
    # Instantiate custom pytorch dataset

    # pretrained data transforms so data is useable by pretained model
    pretrained_transforms = transforms.Compose([transforms.Resize(image_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # data transforms for custom model
    custom_data_transforms = transforms.Compose([transforms.Resize(image_size),
                                                 transforms.ToTensor()])

    full_dataset = StanfordDogs.StanfordDogsDataset(data_dir, custom_data_transforms)

    def make_train_test_data(data_set, split_ratio):
        # Test size based on train size
        # Create train and test datasets
        train_size = int(split_ratio * len(data_set))
        test_size = len(data_set) - train_size

        # Add seed to random_split() to get reproducible results, generator=torch.Generator().manual_seed(42)
        # random_split() creates non-overlapping data
        train_data, test_data = torch.utils.data.dataset.random_split(data_set, [train_size, test_size])

        return train_data, test_data

    train_data, test_data = make_train_test_data(full_dataset, split_ratio)
    return train_data, test_data



def make_data_loaders(train_dataset, test_dataset, batch_size):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def train(model, dataloader, loss_fn, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_fn, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(dataloader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images, labels) in enumerate(dataloader):
            print(images.shape, labels.shape)
            loss = train_batch(images, labels, model, optimizer, loss_fn)
            example_ct +=  len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(model, loss, example_ct, epoch)

def train_batch(images, labels, model, optimizer, loss_fn):
    images = images.to(device)
    labels = labels.to(device)
    model.cuda()
    # Forward pass ➡
    outputs = model(images)
    loss = loss_fn(outputs, labels)

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def train_log(model, loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    wandb.watch(model)
    print(f"\nLoss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def test(model, test_loader):
    print("--------Evaluating model--------")
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {100 * correct / total}%")

        wandb.log({"test_accuracy": correct / total})

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")
    torch.save(model.state_dict(), "model_state_dict.pth")

# Build, train and analyze the model with the pipeline
pet_breed_model = model_pipeline(config)