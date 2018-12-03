import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def readAllImages(dirPath):
	person_wise = {}
	for path in tqdm(os.listdir(dirPath)):
		if "_051_06" in path or "_051_08" in path:
			img = cv2.resize(np.asarray(Image.open(os.path.join(dirPath, path)).convert('RGB'), dtype=np.float32), (150, 150))
			person_id = int(path.split('_')[0])
			person_wise[person_id] = person_wise.get(person_id, []) + [img]
	return person_wise


def personsplit(person_wise, split_ratio=0.25, target_resolution=(48, 48), num_train=100, no_pool=False):
	shuffled_indices = np.random.choice(person_wise.keys(), len(person_wise.keys()), replace=False)
	train_indices = shuffled_indices[:num_train]
	test_indices  = shuffled_indices[num_train:]

	# Make splits for labelled/unlabelled data
	split_data = {}
	for i in range(len(train_indices)):
		indices = np.arange(len(person_wise[train_indices[i]]))
		if train_indices[i] in split_data:
			split_data[train_indices[i]]  = np.concatenate((split_data[train_indices[i]], np.array(person_wise[train_indices[i]])[indices]))
		else:
			split_data[train_indices[i]]  = np.array(person_wise[train_indices[i]])[indices]

	# Contruct siamese-compatible data
	def mix_match_data(data, base, resize=False):
		data_X_left, data_X_right, data_Y = [], [], []
		for i in tqdm(range(len(data))):
			# All images of that person
			for i_sub in base[data[i]]:
				if resize:
					i_sub_smaller = cv2.resize(i_sub, target_resolution)
				else:
					i_sub_smaller = i_sub
				for j in range(i, len(data)):
					# All images of that person
					for j_sub in base[data[j]]:
						if resize:
							j_sub_smaller = cv2.resize(j_sub, target_resolution)
						else:
							j_sub_smaller = j_sub
						data_X_left.append(i_sub_smaller)
						data_X_right.append(j_sub_smaller)
						if i == j:
							data_Y.append([1])
						else:
							data_Y.append([0])
		return (data_X_left, data_X_right), data_Y

	train_X, train_Y = mix_match_data(train_indices, split_data)
	# TODO : Use only the test data that they used (for consistent comparison)
	# test_X, test_Y   = mix_match_data(test_indices, person_wise, resize=True)

	# Free up RAM
	del person_wise

	# return (train_X, train_Y), (pool_X, pool_Y), (test_X, test_Y)
	train_X = np.swapaxes(np.array([train_X[0][:1000], train_X[1][:1000]]), 0, 1)
	train_Y = np.array(train_Y[:1000])
	return (train_X, train_Y)


def generatorFeaturized(X, Y, batch_size, featurize=None, resize_res=None):
	X_left, X_right, Y_send = [], [], []
	while True:
		for i in range(0, len(X), batch_size):
			x_left  = np.array(X[0][i: i + batch_size])
			x_right = np.array(X[1][i: i + batch_size])
			Y       = np.array(Y[i: i + batch_size])
			# De-bias data
			Y_flat  = np.stack([y[0] for y in Y])
			pos = np.where(Y_flat == 1)[0]
			neg = np.where(Y_flat == 0)[0]
			# Don't train on totally biased data
			minSamp = np.minimum(len(pos), len(neg))
			if minSamp == 0:
				continue
			selectedIndices = np.concatenate((np.random.choice(pos, minSamp, replace=False), np.random.choice(neg, minSamp, replace=False)), axis=0)
			Y = Y[selectedIndices]
			x_left, x_right = x_left[selectedIndices], x_right[selectedIndices]
			# Resize, if asked to
			if resize_res:
				x_left, x_right = resizeImages([x_left, x_right], resize_res)
			# Featurize, if asked to
			if featurize:
				x_left  = featurize.process(x_left)
				x_right = featurize.process(x_right)
			if len(Y_send) > 0:
				X_left = np.concatenate((X_left, x_left), axis=0)
				X_right = np.concatenate((X_right, x_right), axis=0)
				Y_send = np.concatenate((Y_send, Y), axis=0)
			else:
				X_left = np.copy(x_left)
				X_right = np.copy(x_right)
				Y_send = np.copy(Y)
			if len(Y_send) >= batch_size:
				yield ([X_left, X_right], Y_send)
				X_left, X_right, Y_send = [], [], []


def resizeImages(images, resize_res):
	resized_left  = [cv2.resize(image, resize_res) for image in images[0]]
	resized_right = [cv2.resize(image, resize_res) for image in images[1]]
	return [np.array(resized_left), np.array(resized_right)]
