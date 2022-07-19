import torch
import torch.utils.data
import torchvision
import torch.nn as nn
from torchvision import transforms

##### Custom imports #####
import image_utils
from data_loader.StanfordDogsDataset import StanfordDogsDataset
##########################

# Data & Dataloader stuff
DATA_DIR = r"C:\Users\Cameron\Documents\python projects\dog classification\data\stanford_dataset\images"

pretrained_transforms = transforms.Compose([transforms.Resize((32, 32)),
											transforms.RandomHorizontalFlip(),
											transforms.ToTensor(),
											transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
											])

basic_transforms = transforms.Compose([transforms.Resize((256, 256)),
									   transforms.ToTensor()])

dataset = StanfordDogsDataset(DATA_DIR, basic_transforms)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=36, shuffle=True)
breed_names = dataset.breed_names
num_of_classes = len(dataset.breed_names)


img_batch, label = next(iter(data_loader))

label_names = [breed_names[idx] for idx in label]

image_utils.show_batch_images(img_batch, label_names)


# Initialise pretrained model
efficientnet_model = torchvision.models.efficientnet_b0(pretrained=True)
print(efficientnet_model)

# Final output layer: "(1): Linear(in_features=1280, out_features=1000, bias=True)"
# First input layer: "(0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)"

## Need to change the final fully connected layer.

efficientnet_model.classifier[1] = nn.Linear(in_features=1280, out_features=120, bias=True)

# Turn off gradient computations
with torch.no_grad():
	efficientnet_model.eval()
	out_data = efficientnet_model(img_batch)

	# softmax used to convert vector of numbers into vector of probabilities
	out_data = out_data.softmax(1)

print(out_data.size())
print(out_data[0])
