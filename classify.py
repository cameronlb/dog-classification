import torch
import torch.utils.data
from torchvision import datasets, models, transforms

##### Custom imports #####
import image_utils
from data_loader.StanfordDogsDataset import StanfordDogsDataset
##########

DATA_DIR = r"C:\Users\Cameron\Documents\python projects\dog classification\data\stanford_dataset\images"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Data augmentation and normalization for training

# add in transforms.RandomResizedCrop(224)
pretrained_transforms = transforms.Compose([transforms.Resize((224, 224)),
											transforms.RandomHorizontalFlip(),
											transforms.ToTensor(),
											transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
											])

dataset = StanfordDogsDataset(DATA_DIR, pretrained_transforms)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
label_names = dataset.breed_names

img_batch, label = next(iter(data_loader))

label_names = [label_names[idx] for idx in label]

image_utils.show_batch_images(img_batch, label_names)