import numpy as np 

class SmallDataSet():
	# dataset object consisting of samples from one class
	def __init__(self, rand_state, images):
		"""
			rand_state: numpy random state
		"""
		self.images = images
		self.rand_state = rand_state

	def sample(self, nb_samples):
		perm_idx = self.rand_state.permutation(self.images.shape[0])
		self.images = self.images[perm_idx]

		if nb_samples > len(self.images):
			raise ValueError('nb_samples is greater than size of this class')

		return self.images[:nb_samples]



def get_a_task(dataset_list, n_way, n_shot, nb_test_shot = 0, rand_state = None):
	# Get a batch of inputs and labels from a meta task 

	def one_hot(labels, nb_classes):
		# Returns one hot numpy array
		def _one_hot(x):
			res = [0 for _ in range(nb_classes)]
			res[x] = 1
			return res

		return np.array([_one_hot(l) for l in labels], dtype=float)

	if rand_state:
		meta_set = rand_state.permutation(dataset_list)
	else:
		meta_set = np.random.permutation(dataset_list)

	batch_inputs = []
	batch_labels = []
	batch_inputs_test = []
	batch_labels_test = []

	for i, set_ in enumerate(meta_set[:n_way]):
		samples  = set_.sample(n_shot+nb_test_shot)
		batch_inputs.append(samples[:n_shot])
		batch_labels.extend([i]*n_shot)
		
		if nb_test_shot: 
			batch_inputs_test.append(samples[n_shot:])
			batch_labels_test.extend([i]*nb_test_shot)

	batch_inputs = np.concatenate(batch_inputs, axis=0)
	batch_labels = one_hot(batch_labels, n_way)	

	if nb_test_shot:
		batch_inputs_test = np.concatenate(batch_inputs_test, axis=0)
		batch_labels_test = one_hot(batch_labels_test, n_way)

	return (batch_inputs, batch_labels), (batch_inputs_test, batch_labels_test)


def get_mini_batch(samples, inner_batch_size, rand_state = None):
	batch_inputs, batch_labels = samples 

	if rand_state:
		perm_idx = rand_state.permutation(batch_inputs.shape[0])
	else:
		perm_idx = np.random.permutation(batch_inputs.shape[0])

	batch_inputs = batch_inputs[perm_idx]
	batch_labels = batch_labels[perm_idx]

	return batch_inputs[:inner_batch_size], batch_labels[:inner_batch_size]
