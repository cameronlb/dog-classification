from torch.utils.data.dataset import Dataset


class StanfordDogsDataset(Dataset):
	def __init__(self, data_root, transforms=None):

		self.class_labels = None
		self.images = None

	def __getitem__(self, index):
		# stuff
		return (img, label)

	def __len__(self):
		return count  # of how many examples(images?) you have

if __name__ == '__main__':
	# Test stuff under here
