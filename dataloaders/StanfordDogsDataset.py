from torch.utils.data.dataset import Dataset
from PIL import Image
from dataloaders import data_utils

class StanfordDogsDataset(Dataset):
	def __init__(self, data_root, transforms=None):

		self.df = data_utils.path_to_pd_df(data_root)
		self.class_labels = self.df["label"]
		self.images = self.df["file_path"]
		self.breeds = self.df["breed_name"]
		self.transform = transforms

	def __getitem__(self, index):

		label = self.class_labels[index]
		img = Image.open(self.images[index])
		breed = self.breeds[index]

		# applying transforms
		if self.transform:
			img = self.transform(img)

		return img, label

	def __len__(self):
		return len(self.class_labels)

if __name__ == '__main__':

	DATA_PATH = r"C:\Users\Cameron\Documents\python projects\dog classification\data\stanford_dataset\images"

	stanford_dogs = StanfordDogsDataset(DATA_PATH)

	print(stanford_dogs.breeds)
	print(stanford_dogs.images)
	img, label = stanford_dogs.__getitem__(0)

	img.show()

	print(label)