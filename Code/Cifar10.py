# Import the libraries
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, UpSampling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
import cv2
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

# Assign variables
batch_size = 64
epochs = 20
nb_train_samples = 50000 # 50000 training samples
nb_valid_samples = 10000 # 10000 validation samples
num_classes = 10

# Resize images for pretrained models- Feature Extraction
def load_cifar10_data(img_rows, img_cols):

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()

    # Resize trainging images
    if K.common.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:nb_valid_samples,:,:,:]])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    return X_train, Y_train, X_valid, Y_valid
x_train1,y_train1,x_test1,y_test1 = load_cifar10_data(75, 75)
x_train,y_train,x_test,y_test = load_cifar10_data(32, 32)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

############################## Data-augmentation ######################################

# Data-Augmentation for resnet101
gen = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    horizontal_flip=True, fill_mode="nearest")

test_gen = ImageDataGenerator()
train = gen.flow(x_train)
test = test_gen.flow(x_test)

# Data-Augmentation for inceptionv3
gen1 = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    horizontal_flip=True, fill_mode="nearest")

test_gen = ImageDataGenerator()
train1= gen1.flow(x_train1)
test1 = test_gen.flow(x_test1)

################################## Feature Extraction ######################################

# Feature extract from resnet101
model1= ResNet101(weights='imagenet', include_top=False,input_shape=(32,32,3))
x_train = model1.predict(x_train)
x_test = model1.predict(x_test)
print(x_train.shape)
# Feature extract from Inceptionv3
model2 = InceptionV3(weights='imagenet', include_top=False,input_shape=(75,75,3))
x_train2 = model2.predict(x_train1)
x_test2 = model2.predict(x_test1)
print(x_train2.shape)

# Merge both the predicted outputs
x_train = np.concatenate((x_train, x_train2),axis=3)
x_test = np.concatenate((x_test, x_test2),axis=3)

# Reshape the inputs for feeding into densenet201
x_train = x_train.reshape(x_train.shape[0],64,64,1)
x_test = x_test.reshape(x_test.shape[0],64,64,1)
print(x_train.shape)
print(x_test.shape)

######################################## Classification #########################################

# Classification via Densenet201 pretraioned model
model = Sequential()
model.add(Conv2D(3,(3, 3), activation='relu', input_shape=(64,64,1)))
model.add(MaxPooling2D((2,2)))
dense = DenseNet201(weights='imagenet',include_top=False)
for layer in dense.layers:
    layer.Trainable = False
model.add(dense)
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.summary()
# Compiling the densenet model
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr = 1e-3,momentum = 0.9),metrics=['acc'])

######################################### Run 3 times ###########################################

model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,shuffle = True,validation_data=(x_test,y_test))
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,shuffle = True,validation_data=(x_test,y_test))
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,shuffle = True,validation_data=(x_test,y_test))
# evaluate the performance
score = model.evaluate(x_test,y_test)
print("Testing accuracy",score[1])
