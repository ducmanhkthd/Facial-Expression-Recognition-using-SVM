import tensorflow as tf
import time
import numpy as np
import argparse
import dlib
import cv2
import os
from skimage.feature import hog
from sklearn.svm import SVC, LinearSVC
from parameters import DATASET, TRAINING, NETWORK, VIDEO_PREDICTOR, HYPERPARAMS
from data_loader import load_data
import _pickle as cPickle

window_size = 24
window_step = 6

classifier1, classifier2, c, ran, ker, gam, it = (SVC, LinearSVC, 175, None, 'rbf', 0.001, -1)
my_model = classifier1(random_state=0, kernel=ker)


def load_model():
    model = None
    with tf.Graph().as_default():
        print("loading pretrained model...")
        data, validation = load_data(validation=True)

        # model = SVC(random_state=HYPERPARAMS.random_state, max_iter=HYPERPARAMS.epochs, kernel=HYPERPARAMS.kernel,
        #             decision_function_shape=HYPERPARAMS.decision_function, gamma=HYPERPARAMS.gamma)
        # model.fit(data['X'], data['Y'])
        if os.path.isfile(TRAINING.save_model_path):
            with open(TRAINING.save_model_path, 'rb') as f:
                model = cPickle.load(f)
        else:
            print("Error: file '{}' not found".format(TRAINING.save_model_path))
    return model


def get_landmarks(image, rects, predictor):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")

    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


def sliding_hog_windows(image):
    print("--1.1--")
    hog_windows = []
    for y in range(0, NETWORK.input_size, window_step):
        print("--1.2--")
        for x in range(0, NETWORK.input_size, window_step):
            print("--1.3--")
            window = image[y:y + window_size, x:x + window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                   cells_per_block=(1, 1), visualize=False))
    return hog_windows


def predict(image, model, shape_predictor=None):
    # get landmarks
    if NETWORK.use_landmarks or NETWORK.use_hog_and_landmarks or NETWORK.use_hog_sliding_window_and_landmarks:
        face_rects = [dlib.rectangle(left=0, top=0, right=NETWORK.input_size, bottom=NETWORK.input_size)]
        face_landmarks = np.array([get_landmarks(image, face_rects, shape_predictor)])
        features = face_landmarks
        if NETWORK.use_hog_sliding_window_and_landmarks:
            print("--1--")
            hog_features = sliding_hog_windows(image)
            print("--111--")
            hog_features = np.asarray(hog_features)
            print("--112--")
            face_landmarks = face_landmarks.flatten()
            print("--113--")
            features = np.concatenate((face_landmarks, hog_features))
            print("--114--")
            print(*features, sep=' ')
        else:
            print("--2--")
            hog_features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                  cells_per_block=(1, 1), visualize=True)
            hog_features = np.asarray(hog_features)
            face_landmarks = face_landmarks.flatten()
            features = np.concatenate((face_landmarks, hog_features))

        tensor_image = image.reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
        print("---333---")
        print(*tensor_image, sep=' ')
        print("---444---")
        print(*features.reshape((1, -1)), sep=' ')
        print("---666---")
        predicted_label = model.predict([tensor_image, features.reshape((1, -1))])
        print("---332---")
        return get_emotion(predicted_label[0])
    else:
        print("--3--")
        tensor_image = image.reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
        predicted_label = model.predict(tensor_image)
        return get_emotion(predicted_label[0])
    return None


def get_emotion(label):
    if VIDEO_PREDICTOR.print_emotions:
        print("- Angry: {0:.1f}%\n- Happy: {1:.1f}%\n- Sad: {2:.1f}%\n- Surprise: {3:.1f}%\n- Neutral: {4:.1f}%".format(
            label[0] * 100, label[1] * 100, label[2] * 100, label[3] * 100, label[4] * 100))
    label = label.tolist()
    return VIDEO_PREDICTOR.emotions[label.index(max(label))], max(label)


# parse arg to see if we need to launch training now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="Image file to predict")
args = parser.parse_args()
if args.image:
    if os.path.isfile(args.image):
        model = load_model()
        image = cv2.imread(args.image, 0)
        print("image")
        shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)
        print("shape_predictor")
        start_time = time.time()
        emotion, confidence = predict(image, model, shape_predictor)
        print("emotion")
        total_time = time.time() - start_time
        print("Prediction: {0} (confidence: {1:.1f}%)".format(emotion, confidence * 100))
        print("time: {0:.1f} sec".format(total_time))
    else:
        print("Error: file '{}' not found".format(args.image))
