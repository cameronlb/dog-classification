import copy
import time
import torch
import torch.utils.data
import torchvision
import wandb
from torch import nn, optim
from torchvision import datasets, models, transforms

##### Custom imports #####
import image_utils
from data_loader.StanfordDogsDataset import StanfordDogsDataset
from models import model_utils
from train_test_model import train_test_model
##########

DATA_DIR = r"C:\Users\Cameron\Documents\python projects\dog classification\data\stanford_dataset\images"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = None
num_epochs = 25
batch_size = 32
config = {}

wandb.login()

wandb.init(project="pytorch-dog-breed-classifier",
           entity="cambino",
           config = {"epochs": num_epochs,
                     "batch_size": batch_size,
                     "criterion": "Cross Entropy Loss",
                     "pretrained_name": "Efficient Net",
                     "loss fn": "SGD"})

wandb.run.name = "Efficient Net: ..."
wandb.run.save()

# Data augmentation and normalization for training

######### ADD TRANSFORMS LATER, TEST WITHOUT FIRST #########
## ADD TO TRAIN
#         transforms.RandomResizedCrop(input_size),
#         transforms.RandomHorizontalFlip()

# ADD TO TEST
#        transforms.Resize(input_size),
#        transforms.CenterCrop(input_size)
######### ######### ######### ######### ######### #########
data_transforms = {"train": transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ]),
                   "test": transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])
                   }


##### CREATING DATASETS #####
# Creating train and test sets, pytorch has no easy way to apply transforms to train + test
train_dataset = StanfordDogsDataset(DATA_DIR, data_transforms["train"])
test_dataset = StanfordDogsDataset(DATA_DIR, data_transforms["test"])

label_names = train_dataset.breed_names
num_classes = len(label_names)

indices = torch.randperm(len(train_dataset))
test_size = len(train_dataset) // 4

train_dataset = torch.utils.data.Subset(train_dataset, indices[:-test_size])
test_dataset = torch.utils.data.Subset(test_dataset, indices[:-test_size])


##### CREATE DATALOADERS #####
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

# dict of dataloaders for training and test pass
dataloaders = {
                "train": train_data_loader,
                "test": test_data_loader,
               }

##### VISUALIZE DATASET #####
img_batch, label = next(iter(train_data_loader))
label_names = [label_names[idx] for idx in label]
image_utils.show_batch_images(img_batch, label_names, True)


##### INITIALIZE MODEL #####
model_effecientnet = torchvision.models.efficientnet_b0(pretrained=True)

model_effecientnet, optimizer = model_utils.initialize_model(model_effecientnet, num_classes)

model_effecientnet.to(DEVICE)

##### TRAIN + TEST MODEL #####
wandb.watch(model_effecientnet, log_freq=100)

trained_model, train_history = train_test_model(model_effecientnet, dataloaders, criterion, optimizer, num_epochs,
												device=DEVICE)


##### SAVE MODELS #####
torch.onnx.export(trained_model, img_batch.to(DEVICE), "model.onnx")
torch.save(trained_model.state_dict(), "model_state_dict.pth")