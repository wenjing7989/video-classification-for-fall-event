from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import mymodels
from loadfall import falldata

import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def test(model_name,tvt='test'):
	batch_size = 1
	data = falldata(tvt)
	generator = data.generate(batch_size)
	steps = data.get_steps()//batch_size

	result = model_name.model.evaluate_generator(generator, steps)
	print(result[1])
# test(mdl)

path='./result/'
model_name = 'crnn'
num_cuts = 5
nb_epoch = 150

saved_model = None #'./result/final.hdf5'
batch_size = 1
nb_classes = 2#5

img_size = [224, 224, 3]

#checkpointer = ModelCheckpoint(filepath=path+model_name+\
#	'.{epoch:03d}-{val_loss:.3f}.hdf5', verbose=1, save_best_only=True)

#timestamp = time.time()
#csv_logger = CSVLogger(path+ model_name + str(timestamp) + '.log')

tdata = falldata('train')
vdata = falldata('val')
steps_per_epoch = tdata.get_steps()//batch_size
validation_steps = vdata.get_steps()//batch_size

generator = tdata.generate(batch_size)
val_generator = vdata.generate(batch_size)

mdl = mymodels(nb_classes, model_name, num_cuts, img_size, saved_model)
mdl.model.fit_generator(generator=generator, steps_per_epoch=steps_per_epoch,
	epochs=nb_epoch, #callbacks=[checkpointer, csv_logger],
	validation_data=val_generator, validation_steps=validation_steps, verbose=2)

mdl.model.save(path+'final.hdf5')
