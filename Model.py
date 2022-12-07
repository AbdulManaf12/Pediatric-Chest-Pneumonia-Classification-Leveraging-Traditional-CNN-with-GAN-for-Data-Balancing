from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
import cv2


class MyModel:
    def __init__(self):
        self.IMAGE_SIZE = 148
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), input_shape=(self.IMAGE_SIZE,self.IMAGE_SIZE,1), activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
        self.model.load_weights('static/weights.h5')

    def predict(self, img_path):
        i = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        i = cv2.resize(i, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        i = np.array(i).reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1)
        i = np.array(i).astype('float')/255.0
        prediction = self.model.predict([i])
        return (['NORMAL', 'PNEUMONIA'][int(prediction.round())], prediction[0][0])

