# import tensorflow as tf
# import time
# import numpy as np
# import argparse
# import dlib
# import cv2
# import os
# from skimage.feature import hog
# from sklearn.svm import SVC, LinearSVC
# from parameters import DATASET, TRAINING, VIDEO_PREDICTOR
# from data_loader import load_data
# import _pickle as cPickle
#
# window_size = 24
# window_step = 6
#
# classifier1, classifier2, c, ran, ker, gam, it = (SVC, LinearSVC, 175, None, 'rbf', 0.001, -1)
# my_model = classifier1(random_state=0, kernel=ker)
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#
# def load_model():
#     model = None
#     with tf.Graph().as_default():
#         print("loading pretrained model...")
#         data, validation = load_data(validation=True)
#
#         # model = SVC(random_state=HYPERPARAMS.random_state, max_iter=HYPERPARAMS.epochs, kernel=HYPERPARAMS.kernel,
#         #             decision_function_shape=HYPERPARAMS.decision_function, gamma=HYPERPARAMS.gamma)
#         # model.fit(data['X'], data['Y'])
#         if os.path.isfile(TRAINING.save_model_path):
#             with open(TRAINING.save_model_path, 'rb') as f:
#                 model = cPickle.load(f)
#         else:
#             print("Error: file '{}' not found".format(TRAINING.save_model_path))
#     return model
#
#
# def get_landmarks(image, rects):
#     # this function have been copied from http://bit.ly/2cj7Fpq
#     if len(rects) > 1:
#         raise BaseException("TooManyFaces")
#     if len(rects) == 0:
#         raise BaseException("NoFaces")
#     return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])
#
#
# def sliding_hog_windows(image):
#     print("--1.1--")
#     hog_windows = []
#     for y in range(0, 48, window_step):
#         print("--1.2--")
#         for x in range(0, 48, window_step):
#             print("--1.3--")
#             window = image[y:y + window_size, x:x + window_size]
#             hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
#                                    cells_per_block=(1, 1), visualize=False))
#     return hog_windows
#
#
# def predict(image, model, shape_predictor=None):
#     # get landmarks
#     #     face_rects = [dlib.rectangle(left=0, top=0, right=48, bottom=48)]
#     #     face_landmarks = np.array([get_landmarks(image, face_rects, shape_predictor)])
#     #     features = face_landmarks
#     #     print("--2--")
#     #     hog_features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
#     #                           cells_per_block=(1, 1), visualize=True)
#     #     hog_features = np.asarray(hog_features)
#     #     face_landmarks = face_landmarks.flatten()
#     #     features = np.concatenate((face_landmarks, hog_features))
#
#     # Get hog
#     features = sliding_hog_windows(image)
#
#     # Get landmakr
#     face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
#     face_landmarks = get_landmarks(image, face_rects)
#
#     # Build vector
#     print(face_landmarks.shape)
#     face_landmarks = face_landmarks.flatten()
#     X = face_landmarks
#     print(X.shape)
#     # print(X)
#
#     features = np.array(features).flatten()
#     features = features.reshape(1, 2592)
#     print(features.shape)
#
#     X = np.concatenate((X, features), axis=1)
#
#     tensor_image = X
#     print(tensor_image.shape)
#     print("---333---")
#     print(*tensor_image, sep=' ')
#     print("---444---")
#     print(*features.reshape((1, -1)), sep=' ')
#     print("---666---")
#     # predicted_label = model.predict([tensor_image, features.reshape((1, -1))])
#     predicted_label = model.predict(tensor_image)
#     print("---332---")
#     return get_emotion(predicted_label)
#
#
# def get_emotion(label):
#     if VIDEO_PREDICTOR.print_emotions:
#         print("- Angry: {0:.1f}%\n- Happy: {1:.1f}%\n- Sad: {2:.1f}%\n- Surprise: {3:.1f}%\n- Neutral: {4:.1f}%".format(
#             label[0] * 100, label[1] * 100, label[2] * 100, label[3] * 100, label[4] * 100))
#     label = label.tolist()
#     return VIDEO_PREDICTOR.emotions[label.index(max(label))], max(label)
#
#
# # parse arg to see if we need to launch training now or not yet
# parser = argparse.ArgumentParser()
# parser.add_argument("-i", "--image", help="Image file to predict")
# args = parser.parse_args()
# if args.image:
#     if os.path.isfile(args.image):
#         model = load_model()
#         image = cv2.imread(args.image, 0)
#         print("image")
#         shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)
#         print("shape_predictor")
#         start_time = time.time()
#         emotion, confidence = predict(image, model, shape_predictor)
#         print("emotion")
#         total_time = time.time() - start_time
#         print("Prediction: {0} (confidence: {1:.1f}%)".format(emotion, confidence * 100))
#         print("time: {0:.1f} sec".format(total_time))
#     else:
#         print("Error: file '{}' not found".format(args.image))


# 20052020
#  - Với 1 ảnh đầu vào gọi là img
# - Ta thực hiện trích xuất landmark của khuôn mặt trong đó bằng dlib và flattern nó ra
# - Ta tiếp tục lấy HOG Feauture của khuôn mặt đó
# - Concat theo axis=1 cái flatterned landmark với cái hog để ra feature vector
# - Chuyển FV đó thành tensor nếu cần và predict


import tensorflow as tf
import time
import numpy as np
import argparse
import dlib
import cv2
import os
import _pickle as cPickle
from sklearn.svm import SVC, LinearSVC
from parameters import TRAINING, VIDEO_PREDICTOR
from sklearn.metrics import accuracy_score

window_size = 24
window_step = 6

# classifier1, classifier2, c, ran, ker, gam, it = (SVC, LinearSVC, 175, None, 'rbf', 0.001, -1)
# my_model = classifier1(random_state=0, kernel=ker)
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def load_model():
    model = None
    with tf.Graph().as_default():
        print("loading pretrained model...")
        if os.path.isfile(TRAINING.save_model_path):
            with open(TRAINING.save_model_path, 'rb') as f:
                model = cPickle.load(f)
        else:
            print("Error: file '{}' not found".format(TRAINING.save_model_path))
    return model


def get_landmarks(image, rects, predictor=predictor):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")

    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


def sliding_hog_windows(image):
    hog_windows = []
    for y in range(0, 48, window_step):
        for x in range(0, 48, window_step):
            window = image[y:y + window_size, x:x + window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                   cells_per_block=(1, 1), visualize=False))
    return hog_windows


def predict(image, model, shape_predictor=None):
    # Get hog
    features = sliding_hog_windows(image)

    # Get landmakr
    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
    face_landmarks = get_landmarks(image, face_rects)

    # Build vector
    print(face_landmarks.shape)
    face_landmarks = face_landmarks.flatten()
    X = face_landmarks  # .reshape(136,1)
    print(X.shape)
    # print(X)
    features = np.array(features).flatten()
    features = features.reshape(1, 2592)
    # print(features.shape)
    X = np.concatenate((X, features), axis=1)
    tensor_image = X  # np.expand_dims(X,axis=2) #X.reshape(-1,) #image.reshape([-1, 48,48, 1])
    print(tensor_image.shape)
    print("---333---")
    print(*tensor_image, sep=' ')
    # print("---444---")
    # print(*features.reshape((1, -1)), sep=' ')
    # print("---666---")
    predicted_label = model.predict(tensor_image)
    print("predicted_label",*predicted_label, sep=' ')

    y_pred_2 = model.predict_proba(tensor_image)

    return get_emotion(y_pred_2[0])


def get_emotion(label):
    if VIDEO_PREDICTOR.print_emotions:
        print("test")
        print("- Angry: {0:.1f}%\n- Happy: {1:.1f}%\n- Sad: {2:.1f}%\n- Surprise: {3:.1f}%\n- Neutral: {4:.1f}%".format(
            label[0] * 100, label[1] * 100, label[2] * 100, label[3] * 100, label[4] * 100))
    label = label.tolist()
    print("lable", label)
    return VIDEO_PREDICTOR.emotions[label.index(max(label))], max(label)


# parse arg to see if we need to launch training now or not yet
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="Image file to predict")
args = parser.parse_args()

from skimage.feature import hog

if args.image:
    if os.path.isfile(args.image):
        model = load_model()
        image = cv2.imread(args.image, 0)
        # print("shape_predictor",X)
        start_time = time.time()
        emotion, confidence = predict(image, model, predictor)
        print("emotion")
        total_time = time.time() - start_time
        print("Prediction: {0} (confidence: {1:.1f}%)".format(emotion, confidence * 100))  # confidence * 100
        print("time: {0:.1f} sec".format(total_time))
    else:
        print("Error: file '{}' not found".format(args.image))
