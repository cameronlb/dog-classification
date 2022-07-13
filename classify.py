import torch
import torch.utils.data
import numpy as np
import plotly.express as px

from torchvision import datasets, models, transforms

from data_loader.StanfordDogsDataset import StanfordDogsDataset

####### Plotly for image visualisation


DATA_DIR = r"C:\Users\Cameron\Documents\python projects\dog classification\data\stanford_dataset\images"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Data augmentation and normalization for training
pretrained_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
											transforms.RandomHorizontalFlip(),
											transforms.ToTensor(),
											transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
											])

dataset = StanfordDogsDataset(DATA_DIR, pretrained_transforms)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

img_batch, label = next(iter(data_loader))

def invert_normalisation_transform(image):
	npimg = image.numpy()

	npimg = np.transpose(npimg, (1, 2, 0))
	npimg = ((npimg * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406])
	return npimg

def show_batch_images(img_batch, labels):
	# show image batches and labels returned from dataloader

	# convert to numpy and reorder image channels (batch size, c, h, w) -> (batch size, h, w, c)
	images = []
	for img in img_batch:
		images.append(invert_normalisation_transform(img))


	# img_batch = np.moveaxis(img_batch, 1, -1)

	labels = labels.numpy()

	fig = px.imshow(np.array(images), facet_col=0, binary_string=True,
					labels={'facet_col': 'labels'})

	for i, labels in enumerate(labels):
		fig.layout.annotations[i]['text'] = 'class = %d' % labels

	fig.show()

show_batch_images(img_batch, label)