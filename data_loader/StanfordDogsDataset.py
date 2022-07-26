import torch.utils.data.dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision.transforms import transforms
from torch.nn.functional import one_hot
import time

from data_loader import data_utils
import numpy as np

class StanfordDogsDataset(Dataset):
	def __init__(self, data_root, transforms=None):

		self.df = data_utils.path_to_pd_df(data_root)
		self.breed_names = self.df["breed_name"].unique().tolist()
		self.class_labels = self.df["label"].tolist()
		self.images = self.df["file_path"].tolist()
		self.breeds = self.df["breed_name"].tolist()
		self.one_hot_labels = torch.nn.functional.one_hot(torch.as_tensor(self.class_labels), len(self.breeds))
		self.transforms = transforms
		self.train_flag = None
		self.test_flag = None

	def __getitem__(self, index):

		label = self.class_labels[index]
		img = Image.open(self.images[index])
		breed = self.breeds[index]

		# one image in dataset has 4 channels ??????
		if np.array(img).shape[2] == 4:
			img = img.convert("RGB")
			# print(self.images[index])

		if self.transforms:
			img = self.transforms(img)

		return img, label

	def __len__(self):
		return len(self.class_labels)



if __name__ == '__main__':

	DATA_PATH = r"C:\Users\Cameron\Documents\python projects\dog classification\data\stanford_dataset\images"


	custom_data_transforms = transforms.Compose([transforms.Resize((128, 128)),
												 transforms.ToTensor()])

	stanford_dogs = StanfordDogsDataset(DATA_PATH, custom_data_transforms)

	print(stanford_dogs.one_hot_labels)
	print(stanford_dogs.one_hot_labels[1])

	data_loader = DataLoader(stanford_dogs, batch_size=10, shuffle=False)


	batch_imgs, batch_labels = next(iter(data_loader))

	print(len(stanford_dogs))