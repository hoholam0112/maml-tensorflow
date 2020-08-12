import tensorflow as tf 
import numpy as np
import os
import sys
import argparse
import importlib
import models
from sampling import SmallDataSet, get_a_task, get_mini_batch
from utils import (VariableManeger, interpolate_vars, average_vars, add_vars, 
	subtract_vars, scale_vars, sum_vars)

# Bulid a compatational graph

def run(args):
	# GPU assignment
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

	# set random seed 
	tf.set_random_seed(args.rd)
	rs = np.random.RandomState(args.rd)

	# get model's hyper-parameters
	inner_step_size = args.inner_step_size # step size for inner updates
	meta_step_size = args.meta_step_size # step size for meta updates
	meta_step_size_final = args.meta_step_size_final
	meta_train_step = args.meta_train_step
	inner_train_step = args.inner_train_step
	if inner_train_step < 2:
		raise ValueError('inner_train_step must be greater than or equal to 2')

	eval_train_step = args.eval_train_step
	eval_inner_batch_size = args.eval_inner_batch_size

	n_way = args.n_way 
	n_shot = args.n_shot
	n_test_shot = args.n_test_shot
	n_train_shot = args.n_train_shot
	meta_batch_size = args.meta_batch_size
	inner_batch_size = args.inner_batch_size

	# Display hyper-parameter information
	print('=========================================================')
	print('{}-way {}-shot classification: '.format(n_way, n_shot))
	print('inner step size: {}, meta step size: {}'.format(inner_step_size, meta_step_size))
	print('number of meta train steps: ', meta_train_step)
	print('random seed: ', args.rd)	
	print('=========================================================')

	# Get data 
	dataset = importlib.import_module(args.dataset)
	meta_train_set_list, meta_test_set_list = dataset.get_data(flatten=False, centered=True)
	
	# Build a computational graph
	if args.dataset == 'omniglot':
		model_ = models.OmniglotModel(n_way)
	elif args.dataset == 'miniImageNet':
		model_ = models.MiniImageNet(n_way)
	else:
		raise ValueError('Invalid name for dataset')

	classifier = model_.classifier
	input_ph = tf.placeholder(tf.float32, shape = model_.input_shape)
	label_ph = tf.placeholder(tf.float32, shape = [None,n_way])

	with tf.variable_scope('model'):
		pred = classifier(input_ph, is_training=True, reuse=False)

	likelihood = tf.reduce_max(tf.multiply(pred, label_ph), axis=1)
	nll = tf.negative(tf.reduce_mean(tf.log(likelihood + 1e-10)))

	with tf.name_scope('optimizer'):
		train_op = tf.train.AdamOptimizer(inner_step_size, beta1=0).minimize(nll)


	# Control variables
	with tf.variable_scope('control_variables'):
		curr_meta_step = tf.get_variable('curr_meta_step', [], dtype=tf.int32, initializer=tf.zeros_initializer)
		plus_curr_meta_step = tf.assign(curr_meta_step, curr_meta_step+1)

	global_init_op = tf.global_variables_initializer()
	optim_init_op = tf.variables_initializer(tf.global_variables('optimizer'))

	with tf.Session() as sess:
		vm = VariableManeger(tf.trainable_variables('model'), sess)
		sess.run(global_init_op)

		# Saver 
		saver = tf.train.Saver(var_list = tf.global_variables(), max_to_keep = 1)
		save_path = os.path.join('train_logs', args.dataset, args.save_path)
		
		if not os.path.exists(save_path):
			os.mkdir(save_path)
		
		if args.pretrained:
			print('Restore Model')
			saver.restore(sess, '{}/model.ckpt'.format(save_path))

		train_loss = 0.0
		n_total_train = 0
		total_correct = 0
		n_total_test_examples = 0

		# Train the model 

		i = 0
		while i <= meta_train_step:
			i = sess.run(plus_curr_meta_step)

			frac_done = i/meta_train_step
			cur_meta_step_size = frac_done*meta_step_size_final + (1-frac_done)*meta_step_size

			new_weights = []
			old_weights = vm.export_variables()

			for j in range(meta_batch_size):
				# Sample a random task from task distribution
				train_examples, test_examples = get_a_task(meta_train_set_list, 
					n_way, n_train_shot, nb_test_shot=n_test_shot, rand_state = rs)

				batch_inputs_train, batch_labels_train = train_examples
				batch_inputs_test, batch_labels_test = test_examples

				# Initialize optimizer's variables 
				sess.run(optim_init_op)

				for k in range(inner_train_step):
					mini_batch = get_mini_batch(train_examples, inner_batch_size, rs) 
					mini_batch_inputs, mini_batch_labels = mini_batch

					_, train_loss_ = sess.run([train_op, nll], {input_ph:mini_batch_inputs, label_ph:mini_batch_labels})

				new_weights.append(vm.export_variables()) 

				# Evaluate train accuracy
				inputs = np.concatenate([batch_inputs_train, batch_inputs_test], axis=0)
				pred_value = sess.run(pred, {input_ph:inputs})
				nb_correct = np.sum(np.argmax(pred_value[n_train_shot*n_way:], axis=1) == np.argmax(batch_labels_test, axis=1))
				total_correct += nb_correct
				
				vm.import_variables(old_weights) # Return to initial state
				train_loss += train_loss_

			
			new_weight = average_vars(new_weights)
			vm.import_variables(interpolate_vars(old_weights, new_weight, cur_meta_step_size)) # Meta update
			n_total_test_examples += meta_batch_size*n_way*n_test_shot
			n_total_train += meta_batch_size

			if i%10 == 0:
				print('step_%d, train loss: %.4f, train acc: %.4f'%(i, train_loss/n_total_train, total_correct/n_total_test_examples))
				total_correct = 0
				n_total_test_examples = 0
				train_loss = 0.0
				n_total_train = 0

			if i%1000 == 0:
				saver.save(sess, '{}/model.ckpt'.format(save_path))


			# Evaluate model
			if i%100==0:
				evaluate(sess=sess, variable_manager=vm, dataset_list=meta_test_set_list, nb_test_task=20, n_way=n_way, 
					n_shot=n_shot, n_test_shot=n_test_shot, optim_init_op=optim_init_op, inner_train_step=eval_train_step, 
					inner_batch_size=eval_inner_batch_size, rand_state=rs, train_op=train_op, pred_op=pred, input_ph=input_ph,
					label_ph=label_ph)

		if args.test:
			evaluate(sess=sess, variable_manager=vm, dataset_list=meta_test_set_list, nb_test_task=10000, n_way=n_way, 
					n_shot=n_shot, n_test_shot=n_test_shot, optim_init_op=optim_init_op, inner_train_step=eval_train_step, 
					inner_batch_size=eval_inner_batch_size, rand_state=rs, train_op=train_op, pred_op=pred, input_ph=input_ph, 
					label_ph=label_ph, show_progress=True)



def evaluate(sess, variable_manager, dataset_list, nb_test_task, n_way, n_shot, n_test_shot, optim_init_op, 
		inner_train_step, inner_batch_size, rand_state, train_op, pred_op, input_ph, label_ph, show_progress=False):
	total_correct_test = 0
	old_weights = variable_manager.export_variables()

	for j in range(nb_test_task):
		# Progression bar
		if show_progress:
			print("\rComplete: {}/{}".format(j+1,nb_test_task), end="")
			sys.stdout.flush()
			if j == nb_test_task-1:
				print()

		# Sample a random task from task distribution
		train_examples, test_examples = get_a_task(dataset_list, 
		n_way, n_shot, nb_test_shot=n_test_shot, rand_state = rand_state)

		batch_inputs_train, batch_labels_train = train_examples
		batch_inputs_test, batch_labels_test = test_examples

		# Initialize optimizer's variables 
		sess.run(optim_init_op) 

		# Inner train steps
		for _ in range(inner_train_step):
			mini_batch = get_mini_batch(train_examples, inner_batch_size, rand_state) 
			mini_batch_inputs, mini_batch_labels = mini_batch
			sess.run(train_op, {input_ph:mini_batch_inputs, label_ph:mini_batch_labels})

		# count corretectly classified examples
		inputs = np.concatenate([batch_inputs_train, batch_inputs_test], axis=0)
		pred_value = sess.run(pred_op, {input_ph:inputs})
		nb_correct = np.sum(np.argmax(pred_value[n_shot*n_way:], axis=1) == np.argmax(batch_labels_test, axis=1))

		total_correct_test += nb_correct
		variable_manager.import_variables(old_weights)

	# Evaluate train accuracy
	test_accuracy = total_correct_test/(nb_test_task*n_way*n_test_shot)
	print('Test accuracy: %.4f'%(test_accuracy))







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='1-n_way few n_shot classification')
    parser.add_argument('dataset', nargs="?", choices=['omniglot', 'miniImageNet', 'celebA'], 
        help='the name of the dataset you want to run the experiments on')
    parser.add_argument('--meta_train_step', help='number of total meta_train_step', default=100000, type=int)
    parser.add_argument('--rd', help='random_seed', default=1, type=int)
    parser.add_argument('--gpu', help='gpu_number', default=0, type=int)
    parser.add_argument('--n_way', help='n_way', default=5, type=int)
    parser.add_argument('--n_shot', help='n_shot', default=5, type=int)
    parser.add_argument('--n_train_shot', help='n_train_shot', default=15, type=int)
    parser.add_argument('--n_test_shot', help='number of test example per class', default=1, type=int)
    parser.add_argument('--inner_step_size', help='step_size for inner train step', default=1e-3, type=float)
    parser.add_argument('--inner_train_step', help='number of inner train step', default=8, type=int)
    parser.add_argument('--inner_batch_size', help='inner batch size', default=10, type=int)
    parser.add_argument('--meta_step_size', help='step_size for meta train step', default=1.0, type=float)
    parser.add_argument('--meta_step_size_final', help='final_step_size for meta train step', default=0.0, type=float)
    parser.add_argument('--meta_batch_size', help='meta_batch_size', default=5, type=int)
    parser.add_argument('--eval_train_step', help='eval_train_step', default=50, type=int)
    parser.add_argument('--eval_inner_batch_size', help='eval_inner_batch_size', default=20, type=int)
    parser.add_argument('--save_path', help='save_path', default='0', type=str)
    parser.add_argument('--test', help='test trained model', default=False, type=bool)
    parser.add_argument('--pretrained', help='have pretrained model', default=False, type=bool)

    run(parser.parse_args())


