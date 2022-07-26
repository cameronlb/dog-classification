import copy
import time
import torch
import torch.utils.data
import torchvision
import wandb
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

##### Custom imports #####
import image_utils
from data_loader.StanfordDogsDataset import StanfordDogsDataset
from models import model_utils
from models.DogBreedClassifier import DogBreedClassifier
from train_val_model import train_val_model
##########

DATA_DIR = r"C:\Users\Cameron\Documents\python projects\dog classification\data\stanford_dataset\images"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_function = nn.CrossEntropyLoss()
optimizer = None
num_epochs = 50
batch_size = 32
image_size = (224, 224)
learning_rate = 0.001
momentum = None

config = {"epochs": num_epochs,
          "batch_size": batch_size,
          "input_size": image_size,
          "learning_rate": learning_rate,
          "momentum": momentum,
          "loss_function": loss_function._get_name(),
          "optimizer": optimizer,
          "train_data_transforms": None,
          "val_data_transforms": None,
          "data_set": DATA_DIR,
          "train_data_size": None,
          "val_data_size": None
          }


# may need to add random-crops
data_transforms = {"train": transforms.Compose([transforms.Resize(image_size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ]),
                   "val": transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])
                   }


##### CREATING DATASETS #####
# Creating train and val sets, pytorch has no easy way to apply transforms to train + val
train_dataset = StanfordDogsDataset(DATA_DIR, data_transforms["train"])
val_dataset = StanfordDogsDataset(DATA_DIR, data_transforms["val"])

label_names = train_dataset.breed_names
num_classes = len(label_names)

indices = torch.randperm(len(train_dataset))
val_size = len(train_dataset) // 5

train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size])
val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_size:])
# need to make another test set form train set, to test final model

config["train_data_size"] = len(train_dataset)
config["val_data_size"] = len(val_dataset)


##### CREATE DATALOADERS #####
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# dict of dataloaders for training and val pass
dataloaders = {
                "train": train_data_loader,
                "val": val_data_loader,
               }

##### VISUALIZE DATASET #####
img_batch, label = next(iter(train_data_loader))
label_names = [label_names[idx] for idx in label]
train_imgs = image_utils.show_batch_images(img_batch, label_names, True)

##### MODEL INIT #####
model = DogBreedClassifier()
print(model)
#check model params for gradient == true
params_to_update = model.parameters()
print("Params to learn: ")
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print("\t", name)
print("-------------------------------------\n")

optimizer = optim.Adam(params_to_update, lr=learning_rate)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model.to(DEVICE)

config["optimizer"] = str(optimizer)
config["train_data_transforms"] = str(data_transforms["train"])
config["val_data_transforms"] = str(data_transforms["val"])
print("config being used:")
print(config)

##### TRAIN + val MODEL #####
wandb.login()
wandb.init(project="pytorch-dog-breed-classifier",
           entity="cambino",
           config = config)


wandb.watch(model, log_freq=100)

wandb.log({"train examples": train_imgs})

trained_model, train_history = train_val_model(model, dataloaders, loss_function, optimizer, exp_lr_scheduler, num_epochs,
                                               device=DEVICE)


##### SAVE MODELS #####
dummy_input = torch.zeros([batch_size, 3, 224, 224])
torch.onnx.export(trained_model, dummy_input.to(DEVICE), "model.onnx")
model_scripted = torch.jit.script(trained_model)
model_scripted.save("trained_model_scripted.pt")
torch.save(trained_model.state_dict(), "model_state_dict.pth")
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model_state_dict.pth")
wandb.run.log_artifact(artifact)