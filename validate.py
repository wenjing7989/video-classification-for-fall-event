import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix

from loadfall import falldata

# from keras import backend as K
# K.set_learning_phase(0) #set learning phase 0=test, 1=train
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

model = load_model('./result/final.hdf5')
batch_size = 1
data=falldata('test')
steps = np.ceil(data.num//batch_size)
generator = data.generate(batch_size)

#result = model.evaluate_generator(generator, steps)
#print(result[1])

result = model.predict_generator(generator, steps)
prediction = result.argmax(axis=1)
matrix = confusion_matrix(data.true_label[:data.num], prediction)
print('Confusion matrix: ')
print(matrix)

fimg, flabel = [], []
for i in range(data.num):
    if data.true_label[i]!=prediction[i]:
        fimg.append(data.met[i])
        flabel.append(prediction[i])

print('Falsely classified:')
print(fimg, flabel)

print('Accuracy:')
print((data.num - len(flabel))/float(data.num)*100)
# print model.summary()
