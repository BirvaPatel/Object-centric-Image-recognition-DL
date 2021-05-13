# import the necessary librariess
import os
os.environ['CUDA_VISIBLE_DEVICES']="0";
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, MaxPooling2D, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
import numpy as np
import os
import scipy.io
import scipy.misc
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from keras.applications.densenet import DenseNet201
from keras.applications.resnet import ResNet101
import numpy as np
from keras import *
from keras.models import Sequential
from keras.utils import np_utils
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# load the dataset
path = '101_ObjectCategories' #Add the path to the dataset

categories = sorted(os.listdir(path))
ncategories = len(categories)
# Dividing First 30 images are considered as Training and rest as testing

data_train = []
labels_train = []
data_test = []
labels_test = []

for i, category in enumerate(categories):
    counter = 0;
    for f in os.listdir(path + "/" + category):
        ext = os.path.splitext(f)[1]
        fullpath = os.path.join(path + "/" + category, f)
        #print(fullpath)
        label = fullpath.split(os.path.sep)[-2]
        image = cv2.imread(fullpath)
		
        image = cv2.resize(image, (75, 75))
        counter = counter + 1
        if (counter <= 30):      
            data_train.append(image)
            labels_train.append(label)
        else:
            data_test.append(image)
            labels_test.append(label)
            
print ('First 30 images are considered as Training and rest as testing')

x_train = np.array(data_train, dtype="float") / 255.0
x_test = np.array(data_test, dtype="float") / 255.0


lb = LabelBinarizer()

y_train = lb.fit_transform(labels_train)
y_test = lb.fit_transform(labels_test)

print("Data Splitted")

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

####################Data-augmentation ############################

gen = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    horizontal_flip=True, fill_mode="nearest")

test_gen = ImageDataGenerator()
train = gen.flow(x_train)
test = test_gen.flow(x_test)

#####################Feature Extraction ########################

# Feature extract from resnet101
model1 = Sequential()
model1.add(ResNet101(weights='imagenet', include_top=False,input_shape=(75,75,3)))
model1.add(Conv2D(2048,(3, 3), activation='relu'))
model1.summary()
x_train1 = model1.predict(train)
x_test1 = model1.predict(test)
print(x_train1.shape)

# Feature extract from inceptionv3
model2 = Sequential()
model2.add(InceptionV3(weights='imagenet', include_top=False,input_shape=(75,75,3)))
model2.summary()
x_train2 = model2.predict(train)
x_test2 = model2.predict(test)
print(x_train2.shape)
# Combine both features
x_train = np.concatenate((x_train1, x_train2),axis=3)
x_test = np.concatenate((x_test1, x_test2),axis=3)
# Reshape for feed into Densenet201
x_train = x_train.reshape(x_train.shape[0],64,64,1)
x_test = x_test.reshape(x_test.shape[0],64,64,1)
print(x_train.shape)
print(x_test.shape)

######################## Classification #########################

# Classification using Densenet121
model = Sequential()
model.add(Conv2D(3,(3, 3), activation='relu', input_shape=(64,64,1)))
model.add(MaxPooling2D((2,2)))
dense = DenseNet201(weights='imagenet',include_top=False)
for layer in dense.layers:
    layer.Trainable = False
model.add(dense)
model.add(Flatten())
model.add(Dense(102,activation='softmax'))
model.summary()
# Compile the model
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr = 1e-3,momentum = 0.9), metrics=['acc'])

########################## Run 3 times ###########################

model.fit(x_train,y_train, epochs=20,batch_size=32,shuffle=True,
                validation_data=(x_test, y_test))
model.fit(x_train,y_train, epochs=20,batch_size=32,shuffle=True,
                validation_data=(x_test, y_test))
model.fit(x_train,y_train, epochs=20,batch_size=32,shuffle=True,
                validation_data=(x_test, y_test))
# Evaluate the model performance                
score = model.evaluate(x_test, y_test)
print('Test accuracy:', (score[1]*100))
