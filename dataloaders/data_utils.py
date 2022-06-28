import os
import sys
import pandas as pd

def get_breeds_from_folder(data_dir):
	""" Returns a list of all breeds """

	path = os.path.join(sys.path[1], data_dir)
	dir_list = os.listdir(path)

	stanford, tsinghua = None, None
	if 'stanford' in data_dir: stanford = True
	elif 'tsinghua' in data_dir: tsinghua = True
	else: return print("Unrecognized path, exception raised")

	breeds = []
	for folder_name in dir_list:
		if stanford:
			head_tail = folder_name.split('-', 1)
			breeds.append(head_tail[1])

		elif tsinghua:
			head_tail = folder_name.split('-', 2)
			breeds.append(head_tail[2])

	return breeds

def get_breed_images(data_dir, breed):

	path = os.path.join(sys.path[1], data_dir)
	image_folder_path = None
	image_files = None

	print("Searching directory: \"{}\"... with: \"{}\".\n".format(path.split("data", 1)[1],
																						breed))
	for (root, __, __) in os.walk(path, topdown=False):
		if breed in root:
			image_folder_path = root
			print("Found Directory: \"{}\".\n".format(root))

			for (__, __, files) in os.walk(image_folder_path, topdown=False):
				image_files = files

	# maybe change this to return full path using os.join()
	return image_folder_path, image_files


if __name__ == '__main__':
	'''
	data utility functions used for processing data with in the data directory for dataloaders.
	'''
	STANFORD_DATA_DIR = "data\\stanford_dataset\\images"
	TSINGHUA_DATA_DIR = "data\\tsinghua_dataset\\low-resolution"


	tsinghua_breeds = get_breeds_from_folder(TSINGHUA_DATA_DIR)
	stanford_breeds = get_breeds_from_folder(STANFORD_DATA_DIR)
	stanford_breeds.sort()
	tsinghua_breeds.sort()

	# print("TSINGHUA #{} BREEDS: {} \n\n".format(len(tsinghua_breeds), tsinghua_breeds),
	# 	  "STANFORD #{} BREEDS: {} \n".format(len(stanford_breeds), stanford_breeds))

	file_dir, files = get_breed_images(TSINGHUA_DATA_DIR, tsinghua_breeds[0])

	print("Folder path: \"{}\"".format(file_dir))
	print("Files: {}".format(files))


