### This is the CNN training module ###

from keras import backend as K
#K.set_image_dim_ordering('th')
K.set_image_data_format('channels_last')
import numpy as np
import os
from PIL import Image
import time
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.layers import Dense, Activation, Flatten,Dropout
from keras.layers import merge, Input
from keras.models import Model,Sequential
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D, MaxPooling2D
#from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as pltX
data_path='/usr/home/miruna/Desktop/student/Img_dataset/'
img_list = os.listdir(data_path)

img_rows = 224
img_cols = 224

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 3
# number of epochs to train
nb_epoch = 20


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

img_data_list=[]

for img in img_list:
      img_path = data_path+img
      img = image.load_img(img_path, target_size=(224, 224))
      #print(img.size)
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      x = x/255
      #print('Input image shape:', x.shape)
      img_data_list.append(x)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)

num_of_samples = len(img_list)
# Define the number of classes
num_classes = 3


label = np.ones((num_of_samples),dtype='int64')
print(len(img_list))
for i in range(len(img_list)):
    if img_list[i].find('severe')!=-1:
        label[i]=0
    elif img_list[i].find('mild')!=-1:
        label[i]=1
    elif img_list[i].find('none')!=-1:
        label[i]=2

names = ['severe','mild','none']

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(label, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

print(X_train.shape)

# Custom_vgg_model_1
#Training the feature extraction also

image_input = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

model.summary()

last_layer = model.get_layer('block5_pool').output

x= Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
out = Dense(num_classes, activation='softmax', name='output')(x)
custom_vgg_model2 = Model(image_input, out)
#custom_vgg_model2.summary()

# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-3]:
	layer.trainable = False

custom_vgg_model2.summary()

custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

t=time.time()
#t = now()
hist = custom_vgg_model2.fit(X_train, y_train, batch_size=32, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
custom_vgg_model2.save('mymodel_vgg.h5')
custom_vgg_model2.save_weights('myweights_vgg.hdf5',overwrite=True)

# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])