# import libraries
import cv2
import os
import zipfile
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

tf.__version__

from google.colab import drive
drive.mount('/content/drive')

path = '/content/drive/MyDrive/Computer Vision/Datasets/homer_bart_1.zip'
zip_object = zipfile.ZipFile(file=path, mode='r')
zip_object.extractall('./')
zip_object.close()

directory = '/content/homer_bart_1'
files = [os.path.join(directory, f) for f in sorted(os.listdir(directory))]
print(files)

width, height = 128, 128

images = []
classes = []

image.shape

128 * 128 * 3, 128 * 128  # nodes in input layer of NN

for image_path in files:
  #print(image_path)

  try:   # to debug
    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]
  except:
    continue
  image = cv2.resize(image, (width, height))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  cv2_imshow(image)  # we cannot show an image after we've converted it into a vector

  image = image.ravel()   # to turn the matrix of pixels into vector of pixels
  print(image.shape)

  images.append(image)

  image_name = os.path.basename(os.path.normpath(image_path))

  if image_name.startswith('b'):
    class_name = 0
  else:
    class_name = 1

  classes.append(class_name)
  print(class_name)

# when working with grayscale images -> all the values of RGB are the same

images

classes

type(images), type(classes)

x = np.asarray(images)
y = np.asarray(classes)

type(x), type(y)

x.shape

y.shape

x[0].reshape(width, height).shape  # reshaping the vector of image to a matrix to be able to show it

cv2_imshow(x[0].reshape(width, height))

sns.countplot(y)

np.unique(y, return_counts=True)

x[0].max(), x[0].min()

# normalizing the data

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x[0].max(), x[0].min()

# splitting the train/ test set

from sklearn.model_selection import train_test_split

x_tarin, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

x_tarin.shape, y_train.shape

x_test.shape, y_test.shape

# building & training the NN via tenserflow

128 * 128

(16384+2)/2   # estimated num of units in hidden layer

network1 = tf.keras.models.Sequential()
network1.add(tf.keras.layers.Dense(input_shape=(16384,), units=8193, activation='relu'))
network1.add(tf.keras.layers.Dense(units=8193, activation='relu'))
network1.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))   # since it's binary classification, it's better to use sigmoid

network1.summary()

# training the NN

network1.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

history = network1.fit(x_tarin, y_train, epochs=50)

# evaluating the NN

history.history.keys()

plt.plot(history.history['loss']);  # not all 50 epochs are needed

plt.plot(history.history['accuracy']);

x_test, x_test.shape

predictions = network1.predict(x_test)
predictions

# 0/ false = bart(little boy)
# 1/ true = homer (man)

predictions = (predictions > 0.5)  # setting a threshold of 0.5 to group the data into true/false

predictions

y_test   # expected output

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
cm

sns.heatmap(cm, annot=True);

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

# saving & loading the model for when i wanna use it again

model_json = network1.to_json()
with open('network1.json', 'w') as json_file:
  json_file.write(model_json)

from keras.models import save_model
network1_saved = save_model(network1, 'weights1.hdf5')

with open('network1.json') as json_file:
  json_saved_model = json_file.read()
json_saved_model

network1_loaded = tf.keras.models.model_from_json(json_saved_model)
network1_loaded.load_weights('/content/weights1.hdf5')
network1_loaded.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

network1_loaded.summary()

# classifying one single image

x[0].shape

test_image = x_test[34]
cv2_imshow(test_image.reshape(width, height))

# since it's showing a black pic after normalizing, we need to convert it to its original format

test_image = scaler.inverse_transform(test_image.reshape(1, -1))

test_image

cv2_imshow(test_image.reshape(width, height))

network1_loaded.predict(test_image)[0]

if network1_loaded.predict(test_image)[0][0] < 0.5 :
  print('Bart')
else:
  print('Homer')

# feature extractor

files = [os.path.join(directory, f) for f in sorted(os.listdir(directory))]
print(files)

export = 'mouth, pants, shoes, tshirt, shorts, sneakers, class\n'

show_images = False
features = []

for image_path in files:
  #print(image_path)
  try:
    original_image = cv2.imread(image_path)
    (H, W) = original_image.shape[:2]
  except:
    continue

  image = original_image.copy()
  image_features = []
  mouth = pants = shoes = 0  # initializing homer's features
  tshirt = shorts = sneakers = 0 # initializing bart's features

  image_name = os.path.basename(os.path.normpath(image_path))

  if image_name.startswith('b'):
    class_name = 0
  else:
    class_name = 1


    for height in range(0, H):
      for width in range(0, W):
        blue = image.item(height, width, 0)
        green = image.item(height, width, 1)
        red = image.item(height, width, 2)

        if class_name == 1 :
          # homer -> brown mouth
          if (blue >= 95 and blue <= 140 and green >= 160 and green <= 185 and red >= 175 and red <= 200):
            image[height, width] = [0, 255, 255]  # the color that features will be showed in, when detected
            mouth += 1

          # homer -> blue pants
          if (blue >= 150 and blue <= 180 and green >= 98 and green <= 120 and red >= 0 and red <= 90):
            image[height, width] = [0, 255, 255]
            pants += 1

          # homer -> gray shoes
          if height > (H / 2):   # since the shoes are at the bottom of the pictures we dont need to prcess all the pixels
            if (blue >= 25 and blue <= 45 and green >= 25 and green <= 45 and red >= 25 and red <= 45):
              image[height, width] = [0, 255, 255]
              shoes += 1

        else:
          # Bart - orange t-shirt
          if (blue >= 11 and blue <= 22 and green >= 85 and green <= 105 and red >= 240 and red <= 255):
            image[height, width] = [0, 255, 128]
            tshirt += 1

          # Bart - blue shorts
          if (blue >= 125 and blue <= 170 and green >= 0 and green <= 12 and red >= 0 and red <= 20):
            image[height, width] = [0, 255, 128]
            shorts += 1

          # Bart - blue sneakers
          if height > (H / 2):
            if (blue >= 125 and blue <= 170 and green >= 0 and green <= 12 and red >= 0 and red <= 20):
              image[height, width] = [0, 255, 128]
              sneakers += 1

  total_pixels = H * W
  mouth = round((mouth / (total_pixels)) * 100, 9)
  pants = round((pants / (total_pixels)) * 100, 9)
  shoes = round((shoes / (total_pixels)) * 100, 9)

  tshirt = round((tshirt / (total_pixels)) * 100, 9)
  shorts = round((shorts / (total_pixels)) * 100, 9)
  sneakers = round((sneakers / (total_pixels)) * 100, 9)


  image_features.append(mouth)
  image_features.append(pants)
  image_features.append(shoes)

  image_features.append(tshirt)
  image_features.append(shorts)
  image_features.append(sneakers)


  features.append(image_features)

  #print('Homer mouth: %s - Homer pants: %s - Homer shoes: %s' %(image_features[0], image_features[1], image_features[2]))
  #print('Bart tshirt: %s - Bart shorts: %s - Bart sneakers: %s' %(image_features[3], image_features[4], image_features[5]))

  f = (",".join([str(item) for item in image_features]))
  export += f + '\n'

  if show_images == True:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # to be able to show an image, it has to be RGB, not BGR
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    fig, im = plt.subplots(1, 2)  # to show 2 images together
    im[0].axis('off')
    im[0].imshow(original_image)
    im[1].axis('off')
    im[1].imshow(image)
    plt.show()

image_features

features

export

with open('features.csv', 'w') as file:
  for l in export:
    file.write(l)
file.closed

dataset = pd.read_csv('features.csv')
dataset

x = dataset.iloc[:, 0:6].values
x

y = dataset.iloc[:, 6].values
y

# splitting the train/ test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

x_train.shape, y_train.shape

x_test.shape, y_test.shape

# building & training the NN
# 6 + 2 / 2 = 4
# 6 inputs(the features only) -> 4 -> 4 -> 4 -> 1

network2 = tf.keras.models.Sequential()   # sequential cauze the nodes & units are connected to each other
network2.add(tf.keras.layers.Dense(input_shape=(6,), units=4, activation='relu'))
network2.add(tf.keras.layers.Dense(units=4, activation='relu'))
network2.add(tf.keras.layers.Dense(units=4, activation='relu'))
network2.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

network2.summary()

network2.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

history = network2.fit(x_tarin, y_train, epochs=50)

# evaluating the NN

history.history.keys()

plt.plot(history.history['loss'])

plt.plot(history.history['accuracy'])

predictions = network2.predict(x_test)

predictions

predictions = (predictions > 0.5)
predictions

y_test

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
cm

sns.heatmap(cm, annot=True);

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

# saving and loading the model

model_json = network2.to_json()
with open('network2.json', 'w') as json_file:
  json_file.write(model_json)

from keras.models import save_model
network2_saved = save_model(network2, '/content/wights2.hdf5')

network2_loaded = tf.keras.models.model_from_json(json_saved_model)
network2_loaded.load_weights('weights2.hdf5')
network2_loaded.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

network2_loaded.summary()

test_image = x_test[33]
test_image

test_image.shape

test_image = test_image.reshape(1, -1)
test_image.shape

network2_loaded.predict(test_image)

if network2_loaded.predict(test_image)[0][0] < 0.5:
  print('bart')
else:
  print('homer')







