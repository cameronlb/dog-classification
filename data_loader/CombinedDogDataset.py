from torch.utils.data.dataset import Dataset


class CombinedDogDataset(Dataset):
	def __init__(self, ...):

	# stuff

	def __getitem__(self, index):
		# stuff
		return (img, label)

	def __len__(self):
		return count  # of how many examples(images?) you have

if __name__ == '__main__':
	# Test stuff under here
