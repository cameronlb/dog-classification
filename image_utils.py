import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn_image as isns
import seaborn as sns
import plotly.express as px
import plotly.subplots
import plotly.graph_objects as go



def invert_normalisation_transform(image):
	""" function to denormalise image applied via transforms """

	npimg = image.numpy()
	npimg = np.transpose(npimg, (1, 2, 0))

	# specific transform normalise for pretrained
	npimg = ((npimg * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406])
	return npimg

def show_batch_images(img_batch, labels, normalise=None):
	""" show image batches and labels returned from dataloader """

	# convert to numpy and reorder image channels (batch size, c, h, w) -> (batch size, h, w, c)
	images = []
	for img in img_batch:
		if normalise:
			images.append(invert_normalisation_transform(img))
		else:
			np_img = img.numpy()
			np_img = np.transpose(np_img, (1, 2, 0))
			images.append(np_img)

	num_images = np.shape(images)[0]

	# number of images to create rows and cols

	if math.sqrt(num_images) % 2 == 0:
		rows = math.sqrt(num_images)
		cols = math.sqrt(num_images)
	else:
		test_num = int(math.sqrt(num_images))
		while num_images % test_num != 0:
			test_num -= 1
		rows = test_num
		cols = num_images / test_num
	rows = int(rows)
	cols = int(cols)

	fig = plotly.subplots.make_subplots(rows=rows, cols=cols)
	idx = 0
	for i, r in enumerate(range(int(rows))):
		for j, c in enumerate(range(int(cols))):
			fig.add_trace(px.imshow(images[idx]).data[0], row=r+1, col=c+1)
			fig.update_xaxes(title_text=f"{labels[idx]}", row=r+1, col=c+1)
			fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
			idx += 1

	fig.show()
	return fig