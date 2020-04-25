import os


class Dataset:
    name = 'Fer2013'
    train_folder = 'fer2013_features/Training'
    validation_folder = 'fer2013_features/PublicTest'
    test_folder = 'fer2013_features/PrivateTest'
    trunc_trainset_to = -1
    trunc_validationset_to = -1
    trunc_testset_to = -1
    shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'


class Hyperparams:
    random_state = 0
    epochs = 10000
    epochs_during_hyperopt = 500
    kernel = 'rbf'  # 'rbf', 'linear', 'poly' or 'sigmoid'
    decision_function = 'ovr'  # 'ovo' for OneVsOne and 'ovr' for OneVsRest'
    features = "landmarks_and_hog"  # "landmarks" or "hog" or "landmarks_and_hog"
    gamma = 'auto'  # use a float number or 'auto'


class Training:
    save_model = True
    save_model_path = "saved_model.bin"


class VideoPredictor:
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    print_emotions = False
    camera_source = 0
    face_detection_classifier = "lbpcascade_frontalface.xml"
    show_confidence = False
    time_to_wait_between_predictions = 0.5


class Network:
    input_size = 48
    output_size = 7  # 7 cam xuc khuon mat
    activation = 'relu'
    loss = 'categorical_crossentropy'
    use_landmarks = True
    use_hog_and_landmarks = True
    use_hog_sliding_window_and_landmarks = True
    use_batchnorm_after_conv_layers = True
    use_batchnorm_after_fully_connected_layers = False


DATASET = Dataset()
TRAINING = Training()
HYPERPARAMS = Hyperparams()
NETWORK = Network()
VIDEO_PREDICTOR = VideoPredictor()
