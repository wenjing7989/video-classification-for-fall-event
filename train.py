from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import mymodels
from loadfall import falldata

import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'# suppress warning

def test(model_name,tvt='test'):
	data = falldata(tvt)
	video, label = data.load_all()

	result = model_name.model.evaluate(video, label)

	print(result[1])
# test(mdl)

def matrix(model_name,tvt='test'):
	from sklearn.metrics import confusion_matrix
	data = falldata(tvt)
	video, label = data.load_all()

	result = model_name.model.predict(video)
	prediction = result.argmax(axis=1)
	matrix = confusion_matrix(label.argmax(axis=1), prediction)

	return matrix

path='./result/'
model_name = 'crnn'
num_cuts = 5
nb_epoch = 5

saved_model = None #'./result/final.hdf5'
batch_size = 1
nb_classes = 2#5

img_size = [224, 224, 3]

#checkpointer = ModelCheckpoint(filepath=path+model_name+\
#	'.{epoch:03d}-{val_loss:.3f}.hdf5', verbose=1, save_best_only=True)

timestamp = time.time()
csv_logger = CSVLogger(path+ model_name + str(timestamp) + '.log')

tdata = falldata('train')
vdata = falldata('val')

Xtrain, Ytrain = tdata.load_all()
validation = vdata.load_all()

mdl = mymodels(nb_classes, model_name, num_cuts, img_size, saved_model)
mdl.model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=nb_epoch,
	callbacks=[csv_logger], validation_data=validation, verbose=2)

print(test(mdl))
print(matrix(mdl))
mdl.model.save(path+'final.hdf5')
