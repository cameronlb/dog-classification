import numpy as np
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

def show_batch_images(img_batch, labels):
	""" show image batches and labels returned from dataloader """

	# convert to numpy and reorder image channels (batch size, c, h, w) -> (batch size, h, w, c)
	images = []
	for img in img_batch:
		images.append(invert_normalisation_transform(img))

	print(np.shape(images))

	# number of images to create rows and cols
	rows = int(np.shape(images)[0] / 4)
	cols = int(np.shape(images)[0] / 2)
	print(rows, cols)

	fig = plotly.subplots.make_subplots(rows=rows, cols=cols)

	idx = 0
	for i, r in enumerate(range(rows)):
		for j, c in enumerate(range(cols)):
			print(np.shape(images[idx]))
			print(idx)
			fig.add_trace(px.imshow(images[idx]).data[0], row=r+1, col=c+1)
			fig.update_xaxes(title_text=f"{labels[idx]}", row=r+1, col=c+1)
			idx += 1


	# fig = px.imshow(np.array(images), facet_col=0, binary_string=True,
	# 				labels={'facet_col': 'labels'})
	#
	# for i, label in enumerate(labels):
	# 	fig.layout.annotations[i]['text'] = f'class = {label}'
	# 	fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
	# 	fig.update_annotations(yanchor="bottom")
	# 	print(label)

	fig.show()

	return fig