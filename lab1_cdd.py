'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs

'''

from __future__ import print_function
import time
import numpy as np
np.random.seed(1337)  # for reproducibility

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.models import Model
from sklearn.decomposition import PCA
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils





time1=time.time()
batch_size = 128
nb_classes = 10
nb_epoch = 15

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)  #60000*28*28

#X_train = X_train.reshape(60000, 784)
#X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)




#MLP784*600*600*10
model1 = Sequential()
model1.add(Reshape((784,), input_shape=(28,28)))#reshape
model1.add(Dense(600))
model1.add(Activation('relu'))
model1.add(Dropout(0.3))
model1.add(Dense(256,name='dense_2'))
model1.add(Activation('relu',name='acti_2'))
model1.add(Dropout(0.5,name='drop_2'))
model1.add(Dense(10,name='output'))
model1.add(Activation('softmax'))

model1.summary()

model1.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history1 = model1.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))



layer_name_dense='dense_2'

layer_model_dense = Model(input=model1.input,
                                 output=model1.get_layer(layer_name_dense).output)

train_PCA_dense = layer_model_dense.predict(X_train)

#a = train_PCA_acti - train_PCA_drop
#print(a)
train_PCA=train_PCA_dense
print(train_PCA.shape)
test_PCA = layer_model_dense.predict(X_test)
print(test_PCA.shape)

pca = PCA(n_components=2)
X_r = pca.fit(train_PCA).transform(test_PCA)
print(X_r.shape)

target_names=['0','1','2','3','4','5','6','7','8','9']
colors = ['black', 'green', 'darkorange','darksalmon','blue','gray','red','cyan','olive','yellow']
markers=['o','h','+','x','D','d','1','2','3','4']
plt.figure(1)

lw = 2

for color, mark,i, target_name in zip(colors,markers, [0, 1, 2,3,4,5,6,7,8,9], target_names):
    plt.scatter(X_r[y_test == i, 0], X_r[y_test == i, 1], color=color,marker=mark, alpha=.4, lw=lw,linewidths=1,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of MLP')


time2=time.time()
print('time2-time1=:',time2-time1)



#2CNN

model2 = Sequential()
model2.add(Reshape((28,28,1,), input_shape=(28,28)))#reshape
model2.add(Convolution2D(32, 3, 3, border_mode='valid'))
model2.add(Activation('relu'))
model2.add(Convolution2D(32, 3, 3))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Convolution2D(64, 3, 3, border_mode='valid'))
model2.add(Activation('relu'))
model2.add(Convolution2D(64, 3, 3))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(256,name='good'))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))
model2.add(Dense(10))
model2.add(Activation('softmax'))

model2.summary()

# Let's train the model using RMSprop
model2.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model2.fit(X_train, Y_train,batch_size=batch_size,nb_epoch=nb_epoch,
           validation_data=(X_test, Y_test),shuffle=True)



layer_name='good'
intermediate_layer_model = Model(input=model2.input,
                                 output=model2.get_layer(layer_name).output)
train_PCA = intermediate_layer_model.predict(X_train)
print(train_PCA.shape)
test_PCA = intermediate_layer_model.predict(X_test)
print(test_PCA.shape)

#pca = PCA(n_components=2)
X_r = pca.fit(train_PCA).transform(test_PCA)
print(X_r.shape)


target_names=['0','1','2','3','4','5','6','7','8','9']
colors = ['black', 'green', 'darkorange','darksalmon','blue','gray','red','cyan','olive','yellow']
markers=['o','h','+','x','D','d','1','2','3','4']
plt.figure(2)

lw = 2

for color, mark,i, target_name in zip(colors,markers, [0, 1, 2,3,4,5,6,7,8,9], target_names):
    plt.scatter(X_r[y_test == i, 0], X_r[y_test == i, 1], color=color,marker=mark, alpha=.4, lw=lw,linewidths=1,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of CNN')

time3=time.time()
print('time3-time2=:',time3-time2)

plt.show()












