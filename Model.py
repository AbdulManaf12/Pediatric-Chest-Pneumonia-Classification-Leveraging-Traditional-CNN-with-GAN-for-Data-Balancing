import numpy as np
import keras
import pickle
import cv2


class MyModel:
    def __init__(self):
        # self.model = pickle.load(open('./static/model.pkl', 'rb'))['Model']
        self.model = self.get_model()
        # pass

    def predict(self, img_path):
        IMAGE_SIZE = 150
        i = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        i = cv2.resize(i, (IMAGE_SIZE, IMAGE_SIZE))
        i = np.array(i).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        i = np.array(i).astype('float')/255.0
        prediction = self.model.predict([i])
        print(['NORMAL', 'PNEUMONIA'][int(prediction.round())], prediction[0][0])
        return (['NORMAL', 'PNEUMONIA'][int(prediction.round())], prediction[0][0])
    
    def get_model():
        model = keras.models.load_model('archeitecure.json')
        model.load_weights('model-weights.h5')

        return model
