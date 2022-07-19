import copy
import time
import torch
import torch.utils.data
import torchvision
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
num_epochs = 10

# Data augmentation and normalization for training

######### ADD TRANSFORMS LATER, TEST WITHOUT FIRST #########
## ADD TO TRAIN
#         transforms.RandomResizedCrop(input_size),
#         transforms.RandomHorizontalFlip()

# ADD TO TEST
#        transforms.Resize(input_size),
#        transforms.CenterCrop(input_size)
######### ######### ######### ######### ######### #########
data_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])

dataset = StanfordDogsDataset(DATA_DIR, data_transforms)
label_names = dataset.breed_names
num_classes = len(label_names)


##### CREATE DATALOADERS #####
train, test, val = dataset.get_train_test_val(0.7)
train_data_loader = torch.utils.data.DataLoader(train, batch_size=36, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test, batch_size = 36, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val, batch_size = 36, shuffle=True)

# dict of dataloaders for training and test pass
dataloaders = {
                "train": train_data_loader,
                "test": test_data_loader,
                "val": val_data_loader
               }

########## Visualise Dataset ##########
img_batch, label = next(iter(train_data_loader))
label_names = [label_names[idx] for idx in label]
image_utils.show_batch_images(img_batch, label_names, True)


model_effecientnet = torchvision.models.efficientnet_b0(pretrained=True)

updated_pretrained_model, optimizer = model_utils.initialize_model(model_effecientnet, num_classes)

updated_pretrained_model.to(DEVICE)

trained_model, train_history = train_test_model(model_effecientnet, dataloaders, criterion, optimizer, num_epochs,
												device=DEVICE)

print(train_history)

torch.onnx.export(trained_model, "model.onnx")
torch.save(trained_model.state_dict(), "../model_state_dict.pth")