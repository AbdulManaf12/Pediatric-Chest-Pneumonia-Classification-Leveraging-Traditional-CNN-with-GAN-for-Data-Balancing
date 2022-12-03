import tensorflow as tf
import numpy as np
import cv2


class MyModel:
    def __init__(self):
        self.model = tf.keras.models.load_model('./static/model.h5')

    def predict(self, img_path):
        IMAGE_SIZE = 150
        i = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        i = cv2.resize(i, (IMAGE_SIZE, IMAGE_SIZE))
        i = np.array(i).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        i = np.array(i).astype('float')/255.0
        prediction = self.model.predict([i])
        return (['NORMAL', 'PNEUMONIA'][int(prediction.round())], prediction[0][0])