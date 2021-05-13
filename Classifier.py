import git
import numpy as np
import matplotlib.pyplot as plt
import keras
import random
import chumpy as cu
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import cv2

path = 'Enter the path where these images have to be saved'
clone = 'https://bitbucket.org/jadslim/german-traffic-signs'
g = git.Git(path).clone(clone)
np.random.seed(0)

with open('C:\\Users\\HP\\PycharmProjects\\IP\\Traffic Data\\german-traffic-signs\\train.p', 'rb') as f:
    train_data = pickle.load(f)
with open('C:\\Users\\HP\\PycharmProjects\\IP\\Traffic Data\\german-traffic-signs\\valid.p', 'rb') as f:
    valid_data = pickle.load(f)
with open('C:\\Users\\HP\\PycharmProjects\\IP\\Traffic Data\\german-traffic-signs\\test.p', 'rb') as f:
    test_data = pickle.load(f)

print(type(train_data)) # we will get a dictionary which has features and lables

X_train, y_train = train_data['features'], train_data['labels']
#features corresponds to the trainig images in the pix values
X_test, y_test = test_data['features'], test_data['labels']
X_val, y_val = valid_data['features'], valid_data['labels']

plt.imshow(X_train[1000])#Check for an arbitary image

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

# To further proceed just pre-checking whether i have same no. of images and labels and its size
assert (X_train.shape[0] == y_train.shape[0]), 'The no. of images is not equal to no. of lables'
assert (X_val.shape[0] == y_val.shape[0]), 'The no. of images is not equal to no. of lables'
assert (X_test.shape[0] == y_test.shape[0]), 'The no. of images is not equal to no. of lables'

assert(X_train.shape[1:] == (32,32,3)), 'The size of the images are not 32x32x3'
assert(X_val.shape[1:] == (32,32,3)), 'The size of the images are not 32x32x3'
assert(X_test.shape[1:] == (32,32,3)), 'The size of the images are not 32x32x3'

data = pd.read_csv('C:\\Users\\HP\\PycharmProjects\\IP\\Traffic Data\\german-traffic-signs\\signnames.csv')
print(data)

#Displaying first 5 images of each class (total 43)
num_of_samples = []

cols = 5
num_classes = 43

fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(10, 50))
fig.tight_layout() #to prevent overlapping of images

for i in range(cols):
    for j, row in data.iterrows(): #it has two arguements like indices & Series(data value)
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, (len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + '-' + row["SignName"])
            num_of_samples.append(len(x_selected))
     
from keras.preprocessing.image import ImageDataGenerator            
def grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def equalize(image):
    image = cv2.equalizeHist(image) #only accept grayscale image as input
    return image

def preprocessing(image):
    image = grayscale(image)
    image = equalize(image)
    image = image / 255
    return image

X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

X_train = X_train.reshape(34799, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)

datagen = ImageDataGenerator(width_shift_range = 0.1,
                   height_shift_range = 0.1,
                   zoom_range = 0.2,
                   shear_range = 0.1,
                   rotation_range = 10)

datagen.fit(X_train)
batches = datagen.flow(X_train, y_train, batch_size = 20)# method for requesting the new images
X_batch, y_batch = next(batches)

fig, axs = plt.subplots(1, 15, figsize = (20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(32, 32))
    axs[i].axis('off')


print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
y_val = to_categorical(y_val, 43)

def LeNet_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(32, 32, 1), activation='relu', strides= 1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr= 0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = LeNet_model()
print(model.summary())#Show the detailed summary of our Architecture

#model.fit_generator is used to parallely run the image generator along with the model
model.fit_generator(datagen.flow(X_train, y_train, batch_size = 50), steps_per_epoch = 2000, epochs = 10, validation_data = (X_val, y_val), shuffle= 1 )


#Score Evaluation 
score = model.evaluate(X_test, y_test, verbose=0)
print(type(score))
print('Test score is:', score[0])
print('Test accuracy is:', score[1])

import requests
from PIL import Image

url = 'https://c8.alamy.com/comp/G667W0/road-sign-speed-limit-30-kmh-zone-passau-bavaria-germany-G667W0.jpg'
r = requests.get(url, stream=True)
img = Image.open(r.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))

# Preprocessing Test image
img = np.asarray(img)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
plt.imshow(img, cmap=plt.get_cmap('gray'))
print(img.shape)

# Reshape reshape
img = img.reshape(1, 32, 32, 1)

print("predicted sign: " + str(np.argmax(model.predict(img))))
plt.show()
