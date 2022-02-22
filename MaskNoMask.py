import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.utils import shuffle

# path = r"C:\Users\MASSON\Desktop\DatasetMask"

IMAGE_SIZE = [150,150]

def load_data(class_names_labels,IMAGE_SIZE):
    datasets = [r'C:\Users\MASSON\Desktop\DatasetMask\output0\train', r'C:\Users\MASSON\Desktop\DatasetMask\output0\val']
    output = []
    for dataset in datasets:
        images = []
        labels = []
        print("Loading {}".format(dataset))
        for folder in os.listdir(dataset):
            try:
                label = class_names_label[folder]
                for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                    img_path = os.path.join(os.path.join(dataset, folder), file)
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, IMAGE_SIZE)
                    images.append(image)
                    labels.append(label)
            except:
                pass
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')
        output.append((images, labels))
    return output

def loading_data_training(class_names_labels,IMAGE_SIZE):
    dd = load_data(class_names_labels,IMAGE_SIZE)
    (train_images, train_labels), (test_images, test_labels) = dd
    train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
    return (train_images, train_labels), (test_images, test_labels)

def data_preprocessing(train_images,test_images):
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images,test_images

def model_spec(N_CLASSES,ACTIVATION):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        #1 : tf.nn.sigmoid
        #3 : tf.nn.softmax
        tf.keras.layers.Dense(N_CLASSES, activation=ACTIVATION)
        ])
    return model

def train_and_compile(train_images,train_labels,test_images,test_labels,model):
    model.compile(optimizer = tf.keras.optimizers.Adam(), loss = tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
    history = model.fit(train_images,train_labels, batch_size=64, epochs=20, validation_data = (test_images,test_labels))
    return history,model

def train_and_compile3(train_images,train_labels,test_images,test_labels,model):
    train_labels = tf.keras.utils.to_categorical(train_labels,3)
    test_labels = tf.keras.utils.to_categorical(test_labels,3)
    model.compile(optimizer = tf.keras.optimizers.Adadelta(), loss = tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    history = model.fit(train_images,train_labels, batch_size=64, epochs=20, validation_data = (test_images,test_labels))
    return history,model


def evaluate(test_images, test_labels):
    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def perf_evaluation(history,train_images,train_labels,test_images,test_labels):
    model_log = history
    pred = model.predict(train_images)
    matrix = tf.math.confusion_matrix(labels=np.round(train_labels), predictions = np.round(pred))
    matrix = matrix.numpy()
    pred = model.predict(test_images)
    matrix = tf.math.confusion_matrix(labels=np.round(test_labels), predictions = np.round(pred))
    matrix = matrix.numpy()

"""Run the pipeline for 2 classes """
# class_names = ['with_mask','without_mask']
# class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
# (train_images, train_labels), (test_images, test_labels) = loading_data_training(class_names_label,IMAGE_SIZE)
# train_images,test_images = data_preprocessing(train_images,test_images)
# model2 = model_spec(1,tf.nn.sigmoid)
# history,model = train_and_compile(train_images,train_labels,test_images,test_labels,model2)
# model2.save(r"C:\Users\MASSON\Desktop\DatasetMask\modele")

""" Run the pipeline for 3 classes """
# class_names = ['incorrect_mask', 'with_mask','without_mask']
# class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
# (train_images, train_labels), (test_images, test_labels) = loading_data_training(class_names_label,IMAGE_SIZE)
# train_images,test_images = data_preprocessing(train_images,test_images)
# model = model_spec(3,tf.nn.softmax)
# history,model = train_and_compile(train_images,train_labels,test_images,test_labels,model)
# model.save(r"C:\Users\MASSON\Desktop\DatasetMask\modele3class")

""" Load models """

model2 = tf.keras.models.load_model(r"C:\Users\MASSON\Desktop\DatasetMask\modele")
model = tf.keras.models.load_model(r"C:\Users\MASSON\Desktop\DatasetMask\modele3class")

""" TEST  MASK/NO MASK WITH DOWNLOADED IMAGES """
def test_images(model2):

    img = cv2.imread(r'C:\Users\MASSON\Desktop\man_masked.jpg')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,IMAGE_SIZE)
    test_img = model2.predict(np.asarray([img]))

    img2 = cv2.imread(r'C:\Users\MASSON\Desktop\man_nomask.jpg')
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2,IMAGE_SIZE)
    test_img2 = model2.predict(np.asarray([img2]))

    img3 = cv2.imread(r'C:\Users\MASSON\Desktop\child_mask.jpg')
    img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
    img3 = cv2.resize(img3,IMAGE_SIZE)
    test_img3 = model2.predict(np.asarray([img3]))

    import matplotlib.pyplot as plt
    plt.imshow(img3)
    titre = 'Masqué' if test_img3.tolist()[0][0]==0 else 'Non masqué'
    plt.title(titre)
    plt.show()

test_images(model2)
# test_images(model)