import os
import sys
import re
import pandas as pd
import pathlib
from PIL import Image

def path_to_pd(img_dir_path):

	full_path = pathlib.Path(img_dir_path)
	print("Converting data path into pandas data frame: \"{}\"".format(full_path))

	breed_match = re.compile("([a-zA-Z][a-zA-Z].*[a-zA-Z])")

	files = full_path.rglob('*.jpg')

	breed_names = []
	folder_names = []
	file_paths = []
	file_names = []
	# fname is absolute file path
	for fname in files:
		if fname.is_file():

			file_paths.append(fname)
			file_names.append(fname.parts[-1])
			folder_names.append(fname.parts[-2])

			match = re.search(breed_match, fname.parts[-2])
			if match:
				breed_names.append(match.group())

	data_frame = pd.DataFrame({'breed_name': breed_names,
							   'folder_name': folder_names,
							   'file_path' : file_paths,
							   'file_name' : file_names})

	unique_breed_names = data_frame['breed_name'].unique()
	print(unique_breed_names)



	return data_frame

if __name__ == '__main__':
	'''
	data utility functions used for processing data with in the data directory for dataloaders.
	'''

	STANFORD_DATA_DIR_PATH = r"C:\Users\Cameron\Documents\python projects\dog " \
							 "classification\data\stanford_dataset\images"
	TSINGHUA_DATA_DIR_PATH = r"C:\Users\Cameron\Documents\python projects\dog " \
							 r"classification\data\tsinghua_dataset\low-resolution"

	df_tsing = path_to_pd(TSINGHUA_DATA_DIR_PATH)
	df_stan = path_to_pd(STANFORD_DATA_DIR_PATH)

	print(df_tsing)
	print(df_stan["breed_name"])