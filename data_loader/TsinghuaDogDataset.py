from torch.utils.data.dataset import Dataset
from PIL import Image
import data_utils

class TsinghuaDogDataset(Dataset):
	def __init__(self, data_root, transforms=None):
		self.df = data_utils.path_to_pd_df(data_root)
		self.class_labels = self.df["label"]
		self.images = self.df["file_path"]
		self.breeds = self.df["breed_name"]

	def __getitem__(self, index):
		# stuff
		label = self.class_labels[index]
		img = Image.open(self.images[index])
		breed = self.breeds[index]

		return img, label

	def __len__(self):
		return len(self.class_labels)

if __name__ == '__main__':

	DATA_PATH = r"C:\Users\Cameron\Documents\python projects\dog classification\data\tsinghua_dataset\low-resolution"

	tsinghua_dogs = TsinghuaDogDataset(DATA_PATH)

	print(tsinghua_dogs.breeds)
	print(tsinghua_dogs.images)
	img, label = tsinghua_dogs.__getitem__(0)

	img.show()

	print(label)