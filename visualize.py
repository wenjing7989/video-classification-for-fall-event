from keras import backend as K
import seaborn as sns
from keras.models import load_model
from loadfall import falldata
import matplotlib.pyplot as plt

from keras import backend as K
K.set_learning_phase(0) #set learning phase

model = load_model('./result/old/conv3.013-0.500.hdf5')
print(model.summary())

data = falldata('val')
train_set, y = data.load_all()

layer_of_interest=10
intermediate_tensor_function = K.function([model.layers[0].input],[model.layers[layer_of_interest].output])
intermediates = intermediate_tensor_function([train_set[:]])[0]
print len(model.layers)
# pdb; pdb.set_trace()
del model
import matplotlib
colors = list(matplotlib.colors.cnames)

color_intermediates = []
for i in range(len(y)):
    color_intermediates.append(colors[int(y[i,0])])

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
intermediates_tsne = tsne.fit_transform(intermediates.reshape(len(intermediates),-1))
del intermediates

plt.figure(figsize=(8, 8))
plt.scatter(x = intermediates_tsne[:,0], y=intermediates_tsne[:,1], color=color_intermediates)
plt.show()
