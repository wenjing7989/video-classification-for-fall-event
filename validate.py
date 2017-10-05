import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix

from loadfall import falldata

# from keras import backend as K
# K.set_learning_phase(1) #set learning phase
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

num_cuts = 1
model = load_model('./result/crnn.000-1.416.hdf5')
batch_size = 1
data=falldata('train')
steps = np.ceil(data.get_steps()//batch_size)
generator = data.generate(batch_size)

result = model.evaluate_generator(generator, steps)
print(result[1])

result = model.predict_generator(generator, steps)
prediction = result.argmax(axis=1)
#matrix = confusion_matrix(true_label, prediction)
# print model.summary()
