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
from models.DogBreedClassifier import DogBreedClassifier
from train_val_model import train_val_model
##########

DATA_DIR = r"C:\Users\Cameron\Documents\python projects\dog classification\data\stanford_dataset\images"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = None
num_epochs = 30
batch_size = 32
image_size = (224, 224)
learning_rate = 0.0001
momentum = 0.9

config = {"epochs": num_epochs,
          "batch_size": batch_size,
          "input_size": image_size,
          "learning_rate": learning_rate,
          "momentum": momentum,
          "loss_function": criterion._get_name(),
          "optimizer": optimizer,
          "train_data_transforms": None,
          "val_data_transforms": None,
          "data_set": DATA_DIR,
          "train_data_size": None,
          "val_data_size": None
          }


# may need to add random-crops
data_transforms = {"train": transforms.Compose([transforms.RandomResizedCrop(image_size),
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
val_size = len(train_dataset) // 4

train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size])
val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_size:])

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


##### INITIALIZE MODEL #####
##### OLD FIXED FEATURE EXTRACTOR MODEL #####
# model_effecientnet = torchvision.models.efficientnet_b0(pretrained=True)
# model_effecientnet, optimizer = model_utils.initialize_model(model_effecientnet, num_classes)


model_custom = DogBreedClassifier()

##### check model params for gradient == true #####
params_to_update = model_custom.parameters()
print("Params to learn: ")
for name, param in model_custom.named_parameters():
    if param.requires_grad == True:
        print("\t", name)

params_to_update = model_custom.parameters()
optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum)

model_custom.to(DEVICE)

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


wandb.watch(model_custom, log_freq=100)

trained_model, train_history = train_val_model(model_custom, dataloaders, criterion, optimizer, num_epochs,
												device=DEVICE)


##### SAVE MODELS #####
torch.onnx.export(trained_model, img_batch.to(DEVICE), "model.onnx")
model_scripted = torch.jit.script(trained_model)
model_scripted.save("trained_model_scripted.pt")
torch.save(trained_model.state_dict(), "model_state_dict.pth")
artifact = wandb.Artifact("model", type="model")
artifact.add_file("model_state_dict.pth")
wandb.run.log_artifact(artifact)