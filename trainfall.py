import argparse
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import mymodels
from loadfall import loaddata

import time
import os

def parse_args():
	parser = argparse.ArgumentParser(description='Train a network')
	parser.add_argument('--model', dest='model_name', type=str)
	parser.add_argument('--path', dest='path', type=str, default='./result/')

	args = parser.parse_args()
	return args

def train(model_name, path):

	# path='./result/a/'
	# model_name = 'mcnn'
	num_cuts = 5
	nb_epoch = 100

	saved_model = None #'./result/vgg.008-0.195.hdf5'
	batch_size = 32
	nb_classes = 2

	img_size = [224, 224, 3]


	checkpointer = ModelCheckpoint(filepath=path+model_name+\
		'.{epoch:03d}-{val_loss:.3f}.hdf5', verbose=1, save_best_only=True)

	timestamp = time.time()
	csv_logger = CSVLogger(path+ model_name + str(timestamp) + '.log')

	steps_per_epoch = 200//batch_size
	validation_steps = 60//batch_size

	generator = loaddata('train', batch_size)
	val_generator = loaddata('val', batch_size)

	mdl = mymodels(nb_classes, model_name, num_cuts, img_size, saved_model)
	mdl.model.fit_generator(generator=generator, steps_per_epoch=steps_per_epoch, 
		epochs=nb_epoch, callbacks=[checkpointer, csv_logger],
		validation_data=val_generator, validation_steps=validation_steps, verbose=2)

	mdl.model.save(path+'final.hdf5')

if __name__=='__main__':
	args = parse_args()
	print('Called with args:')
	print(args)

	train(args.model_name, args.path)
	#python trainfall.py --model 'mcnn' --path './result/a/'

