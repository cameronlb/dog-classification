import torch
from torch_lr_finder import LRFinder
from torch import nn, optim
from torchvision import transforms

from data_loader.StanfordDogsDataset import StanfordDogsDataset
from models.DogBreedClassifier import DogBreedClassifier

DATA_DIR = r"C:\Users\Cameron\Documents\python projects\dog classification\data\stanford_dataset\images"
image_size = (224, 224)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transforms = transforms.Compose([transforms.Resize(image_size),
										transforms.RandomHorizontalFlip(),
										transforms.ToTensor(),
										transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
										])


##### TRAIN DATASET #####
train_dataset = StanfordDogsDataset(DATA_DIR, data_transforms)
indices = torch.randperm(len(train_dataset))
val_size = len(train_dataset) // 5
train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size])

##### DATALOADER #####
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

model = DogBreedClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0000001)
lr_finder = LRFinder(model, optimizer, criterion, device=DEVICE)
lr_finder.range_test(train_data_loader, end_lr=1, num_iter=500)
lr_finder.plot()
lr_finder.reset()


