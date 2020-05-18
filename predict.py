import tensorflow as tf
import time
import numpy as np
import argparse
import dlib
import cv2
import os
from skimage.feature import hog
from sklearn.svm import SVC, LinearSVC
from parameters import DATASET, TRAINING, VIDEO_PREDICTOR
from data_loader import load_data
from localbinarypatterns import LocalBinaryPatterns
import _pickle as cPickle
from sklearn.metrics import accuracy_score

def load_model_predict():
    model = None
    with tf.Graph().as_default():
        print("loading pretrained model...")
        data, validation, test = load_data(validation=True, test=True)

        if os.path.isfile(TRAINING.save_model_path):
            with open(TRAINING.save_model_path, 'rb') as f:
                model = cPickle.load(f)
        else:
            print("Error: file '{}' not found".format(TRAINING.save_model_path))

    return model


def predict(image, model):
    print("--3--")
    # tensor_image = image.reshape([-1, 48, 48 , 1])
    print(*image, sep=' ')
    predicted_label = model.predict(image)
    print("predicted_label", predicted_label)
    return get_emotion(predicted_label[0])


def get_emotion(label):
    print('-----image-----')
    if VIDEO_PREDICTOR.print_emotions:
        print("- Angry: {0:.1f}%\n- Happy: {1:.1f}%\n- Sad: {2:.1f}%\n- Surprise: {3:.1f}%\n- Neutral: {4:.1f}%".format(
            label[0] * 100, label[1] * 100, label[2] * 100, label[3] * 100, label[4] * 100))
    label = label.tolist()
    return VIDEO_PREDICTOR.emotions[label.index(max(label))], max(label)


def evaluate(model, X, Y):
    predicted_Y = model.predict(X)
    accuracy = accuracy_score(Y, predicted_Y)
    return accuracy


# parse arg to see if we need to launch training now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="Image file to predict")
args = parser.parse_args()
if args.image:
    if os.path.isfile(args.image):
        model = load_model_predict()
        image = cv2.imread(args.image, 0)
        print("image")
        start_time = time.time()
        emotion, confidence = predict(image, model)
        print("emotion")
        total_time = time.time() - start_time
        print("Prediction: {0} (confidence: {1:.1f}%)".format(emotion, confidence * 100))
        print("time: {0:.1f} sec".format(total_time))
    else:
        print("Error: file '{}' not found".format(args.image))
