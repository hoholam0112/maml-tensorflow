import pickle
import numpy as np
import matplotlib.pyplot as plt
import os 
from sampling import SmallDataSet, get_a_task


def get_data(flatten=False, centered=False, rand_state=np.random.RandomState(1)):
	path = '/home/mlg/miniImagenet'
	with open(os.path.join(path, 'miniimagenet.p'), 'rb') as file:
		dataset = pickle.load(file)

	def get_dataset_list(dataset_):
		dataset_list = []
		for images in dataset_:
			images = images/255.

			if flatten:
				images = np.reshape(images, [images.shape[0], -1])

			if centered:
				images = images*2-1

			dataset_list.append(SmallDataSet(rand_state, images))

		return dataset_list

	train_set_list = get_dataset_list(dataset['train'])
	val_set_list = get_dataset_list(dataset['val'])
	test_set_list = get_dataset_list(dataset['test'])

	return train_set_list, test_set_list



if __name__ == '__main__':
	meta_train_set, meta_test_set = get_data(flatten=False, centered=True)

	print(len(meta_test_set))
	print(len(meta_train_set))

	print(meta_test_set[0].images.shape)
	print(meta_train_set[0].images.shape)

	n_way = 5
	n_shot = 5
	nb_test_shot = 3

	samples_train, samples_test = get_a_task(meta_train_set, n_way, n_shot, nb_test_shot)
	batch_inputs, batch_labels = samples_train
	batch_inputs_test, batch_labels_test = samples_test

	plt.figure()
	for i in range(n_way*n_shot):
		plt.subplot(n_way,n_shot,i+1)
		plt.imshow((batch_inputs[i]+1)/2)

	plt.figure()
	for i in range(n_way*nb_test_shot):
		plt.subplot(n_way,nb_test_shot,i+1)
		plt.imshow((batch_inputs_test[i]+1)/2)


	# print(meta_test_set[100].images.shape)
	# print(meta_train_set[100].images.shape)
	# print(batch_labels)
	# print(batch_labels_test)

	plt.show()
