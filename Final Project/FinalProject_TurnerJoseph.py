import pandas as pd
import os
import cv2
from matplotlib import pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection._split import train_test_split
from keras.utils.np_utils import to_categorical
from keras.engine.sequential import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import SGD, adadelta, adagrad, adam
from sklearn.metrics import classification_report
from keras.losses import categorical_crossentropy
import random
from numpy.random.mtrand import randint


# path to images
dataset_path = 'cats/'
test_path = 'test/'

# network vars
epochs = 10
batch_size = 32

le = preprocessing.LabelEncoder()

# image vars
image_width, image_height = 128, 128
image_depth = 3
input_shape = (image_width, image_height, image_depth)

# image to predict
image_path = test_path + '96753206_3057061907718284_8582682320677371904_n.jpg'
#image_path = test_path + '46499754_24589.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (image_width, image_height))
image = image / 255.0

# display image
plt.imshow(image)
# plt.show()
plt.savefig('image_plot.png')

# reshape image
image = np.array(image).reshape(-1, image_width,
                                image_height, image_depth)

# exit()


def get_all_breeds():
    cats = pd.read_csv('cats.csv', index_col=0, usecols=[0, 1, 8])
    images, breeds, data = [], [], []
    class_folders = os.listdir(dataset_path)
    for class_name in class_folders:
        print('processing folder: ' + class_name)
        image_list = os.listdir(dataset_path + class_name)
        count = 0
        for image_name in image_list:
            id = image_name.split('_')
            # print('Processing ' + class_name + ' #' + id[1])
            cat = cats.loc[cats['id'] == int(id[0])]
            # print(cat['id'].values[0])
            if not cat.empty:
                if not cat['breed'].values[0] == class_name:
                    print(class_name + '/' + image_name +
                          ' listed as ' + cat['breed'].values[0])
                else:
                    breeds.append(cat['breed'].values[0])
                    image = cv2.imread(dataset_path+class_name+'/' +
                                       image_name)
                    image = cv2.resize(image, (image_width, image_height))
                    images.append(image)
                    count += 1
        data.append([class_name, count])
    return images, breeds, data


def get_breed_data(breed):
    breed_path = dataset_path + breed + '/'
    cats = pd.read_csv('cats.csv', index_col=0, usecols=[0, 1, 4, 5, 6, 8])
    images, ages, genders, sizes = [], [], [], []
    image_list = os.listdir(breed_path)
    for image_name in image_list:
        id = image_name.split('_')
        # print('Processing ' + class_name + ' #' + id[1])
        cat = cats.loc[cats['id'] == int(id[0])]
        # print(cat['id'].values[0])
        if not cat.empty:
            if not cat['breed'].values[0] == breed:
                print(breed + '/' + image_name +
                      ' listed as ' + cat['breed'].values[0])
            else:
                if cat['gender'].values[0] == 'Male' or cat['gender'].values[0] == 'Female':
                    ages.append(cat['age'].values[0])
                    genders.append(cat['gender'].values[0])
                    sizes.append(cat['size'].values[0])
                    image = cv2.imread(breed_path +
                                       image_name)
                    image = cv2.resize(image, (image_width, image_height))
                    images.append(image)
    return images, ages, genders, sizes


def process_images(images):
    images = np.array(images).reshape(-1, image_width,
                                      image_height, image_depth)  # reshape image array
    images = images / 255  # flatten image array
    return images


def process_data(data):
    labels = le.fit_transform(data)
    classes = le.classes_
    count = len(classes)
    return np.array(data), labels, classes, count, to_categorical(labels, count)


# the following model is based on 9_keras_mnist_cnn.py by Feng Jiang
def get_feng_jiang(count):
    print('[INFO] constructing model...')
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), padding="same",
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(count, activation='softmax'))
    print('[INFO] compiling model...')
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd,
                  metrics=["accuracy"])
    return model


# the following model is based on
# Classify butterfly images with deep learning in Keras
# by Bert Carremans
# https://towardsdatascience.com/classify-butterfly-images-with-deep-learning-in-keras-b3101fe0f98
def get_bert_carremans(count):
    print('[INFO] constructing model...')
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same",
                     input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(
        1, 1), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=count, activation='sigmoid'))
    print('[INFO] compiling model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])
    return model


def get_joe_turner(count):
    print('[INFO] constructing model...')
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(count, activation='softmax'))
    print('[INFO] compiling model...')
    model.compile(loss="categorical_crossentropy", optimizer='adagrad',
                  metrics=["accuracy"])
    return model


# gather breed data
print('[INFO] processing dataset...')
images, breeds, data = get_all_breeds()

# output number of images per breed
print('[INFO] dataset composition:')
for datum in data:
    print(datum[0] + ': ' + str(datum[1]))

# process images
images = process_images(images)

# process breed data
breeds, breed_labels, breed_classes, breed_count, breed_categories = process_data(
    breeds)

# BEGIN BREED PREDICTION

# split the breed data
breed_train_x, breed_test_x, breed_train_y, breed_test_y = train_test_split(
    images, breed_categories)
print('[INFO] train test split done')

# construct breed model
model = get_feng_jiang(breed_count)

# train the model
print('[INFO] training model...')
H = model.fit(breed_train_x, breed_train_y, validation_data=(breed_test_x, breed_test_y),
              epochs=epochs, batch_size=batch_size)

# evaluate the network
print("[INFO] evaluating network...")
breed_predictions = model.predict(breed_test_x, batch_size=batch_size)
breed_report = classification_report(breed_test_y.argmax(axis=1),
                                     breed_predictions.argmax(axis=1))
print(breed_report)

predicted_breeds = model.predict_classes(image)
predicted_breed = breed_classes[predicted_breeds[0]]
print('[INFO] predicted breed: ' + predicted_breed)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy - Breed")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
# plt.show()
plt.savefig('breed_tla_plot.png')

del breed_report, breed_predictions, H, model, breed_train_x, breed_test_x, breed_train_y, breed_test_y, breeds, breed_classes, breed_count, breed_categories, images, data

# gather info for predicted breed
print('[INFO] processing ' + predicted_breed + '...')
breed_images, breed_ages, breed_genders, breed_sizes = get_breed_data(
    predicted_breed)

# process predicted breed images
print('[INFO] processing ' + predicted_breed + ' images')
breed_images = process_images(breed_images)

# BEGIN AGE PREDICTION

# process age data
print('[INFO] processing ' + predicted_breed + ' ages')
ages, age_labels, age_classes, age_count, age_categories = process_data(
    breed_ages)

# split data
age_train_x, age_test_x, age_train_y, age_test_y = train_test_split(
    breed_images, age_categories)
print('[INFO] train test split done')

# construct age model
model = get_joe_turner(age_count)

# train age model
print('[INFO] training model...')
H = model.fit(age_train_x, age_train_y, validation_data=(age_test_x, age_test_y),
              epochs=epochs, batch_size=batch_size)

# evaluate the age network
print("[INFO] evaluating network...")
age_predictions = model.predict(age_test_x, batch_size=batch_size)
age_report = classification_report(age_test_y.argmax(axis=1),
                                   age_predictions.argmax(axis=1))
print(age_report)

# predict age
predicted_ages = model.predict_classes(image)
predicted_age = breed_ages[predicted_ages[0]]
print('[INFO] predicted age: ' + predicted_age)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy - Age")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
# plt.show()
plt.savefig('gender_tla_plot.png')

del age_report, age_predictions, H, model, age_train_x, age_test_x, age_train_y, age_test_y, ages, age_classes, age_count, age_categories

# BEGIN GENDER PREDICTION

# process gender data
print('[INFO] processing ' + predicted_breed + ' genders')
genders, gender_labels, gender_classes, gender_count, gender_categories = process_data(
    breed_genders)

# split gender data
gender_train_x, gender_test_x, gender_train_y, gender_test_y = train_test_split(
    breed_images, gender_categories)
print('[INFO] train test split done')

# construct gender model
model = get_bert_carremans(gender_count)

# train gender model
print('[INFO] training model...')
H = model.fit(gender_train_x, gender_train_y, validation_data=(gender_test_x, gender_test_y),
              epochs=epochs, batch_size=batch_size)

# evaluate the gender network
print("[INFO] evaluating network...")
gender_predictions = model.predict(gender_test_x, batch_size=batch_size)
gender_report = classification_report(gender_test_y.argmax(axis=1),
                                      gender_predictions.argmax(axis=1))
print(gender_report)

# predict gender
predicted_genders = model.predict_classes(image)
predicted_gender = breed_genders[predicted_genders[0]]
print('[INFO] predicted gender: ' + predicted_gender)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy - Gender")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
# plt.show()
plt.savefig('age_tla_plot.png')

del gender_report, gender_predictions, H, model, gender_train_x, gender_test_x, gender_train_y, gender_test_y, genders, gender_classes, gender_count, gender_categories
