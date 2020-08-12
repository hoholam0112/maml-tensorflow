import numpy as np
import matplotlib.pyplot as plt
import os 
import pickle
from sampling import SmallDataSet, get_a_task

def get_data(flatten = False, centered=False, rand_state=np.random.RandomState(1)):
	# Get omniglot dataset
	path = '/home/mlg/omniglot'
	with open(os.path.join(path, 'omniglot.p'), 'rb') as file:
		dataset = pickle.load(file)

	dataset_0 = dataset['0']
	dataset_90 = dataset['90']
	dataset_180 = dataset['180']
	dataset_270 = dataset['270']

	all_classes = [e for e in range(len(dataset_0))]
	nb_classes = len(all_classes)

	perm_labels = rand_state.permutation(nb_classes)
	train_labels = perm_labels[:1200]
	test_labels = perm_labels[1200:]

	train_set_list = []
	test_set_list = []

	for label in train_labels:
		train_set_list.append(SmallDataSet(rand_state, dataset_0[label]))
		train_set_list.append(SmallDataSet(rand_state, dataset_90[label]))
		train_set_list.append(SmallDataSet(rand_state, dataset_180[label]))
		train_set_list.append(SmallDataSet(rand_state, dataset_270[label]))

	for label in test_labels:
		test_set_list.append(SmallDataSet(rand_state, dataset_0[label]))

	def process(dataset_list):
		# Scale dataset between 0 and 1
		for dataset in dataset_list:
			dataset.images = 1.0 - dataset.images/255.

			if flatten:
				dataset.images = np.reshape(dataset.images, [dataset.images.shape[0], -1])

			if centered:
				dataset.images = dataset.images*2-1

		dataset_list = rand_state.permutation(dataset_list)

		return dataset_list

	return process(train_set_list), process(test_set_list)


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
		plt.imshow((batch_inputs[i][:,:,0]+1)/2, cmap='gray')

	plt.figure()
	for i in range(n_way*nb_test_shot):
		plt.subplot(n_way,nb_test_shot,i+1)
		plt.imshow((batch_inputs_test[i][:,:,0]+1)/2, cmap='gray')


	# print(meta_test_set[100].images.shape)
	# print(meta_train_set[100].images.shape)
	# print(batch_labels)
	# print(batch_labels_test)

	plt.show()
