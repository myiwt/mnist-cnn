from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import Sequential
import keras
from sklearn.metrics import log_loss
import pickle, os
from scipy.stats import ttest_ind


def load_mnist():
    """
    Downloads the MNIST dataset and conducts pre-processing to prepare the data for model input.

    """
    # load dataset
    # Source: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
    (raw_train_X, raw_train_y), (raw_test_X, raw_test_y) = mnist.load_data()
    print("MNIST data loaded\nTrain data size: {}, test data size: {}".format(len(raw_train_X), len(raw_test_X)))
    print("Train data shape: {}, Test data shape: {}".format(raw_train_X.shape, raw_test_X.shape))

    # Print some samples of the training data
    # Based on a source: https://www.askpython.com/python/examples/load-and-plot-mnist-dataset-in-python
    print("Samples of train data")
    print("From left to right, up to down, the data labels for these samples are: {}".format(raw_train_y[0:9]))
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(raw_train_X[i], cmap=plt.get_cmap('gray'))
    # show the figure
    plt.show()

    # Reshape data
    train_X_reshaped = raw_train_X.reshape((raw_train_X.shape[0], raw_train_X.shape[1], raw_train_X.shape[2], 1))
    test_X_reshaped = raw_test_X.reshape((raw_test_X.shape[0], raw_test_X.shape[1], raw_test_X.shape[2], 1))

    # Normalise data
    train_X_reshaped = train_X_reshaped.astype('float32')
    test_X_reshaped = test_X_reshaped.astype('float32')
    train_X_reshaped_scaled = train_X_reshaped / 255.0
    test_X_reshaped_scaled = test_X_reshaped / 255.0

    # Transform raw labels into vectors with binary values
    lb = LabelBinarizer()
    lb.fit(raw_train_y)
    train_y_labels = lb.transform(raw_train_y)
    test_y_labels = lb.transform(raw_test_y)

    return train_X_reshaped_scaled, train_y_labels, test_X_reshaped_scaled, test_y_labels

def build_train_model(train_X, train_y, input_shape, n_classes=10, batch_size=128, epochs=3):
    """
    Builds and trains a model for MNIST classification

    @param train_X: The MNIST train dataset
    @param train_y: The MNIST train labels
    @param input_shape: The input train data shape
    @param n_classes: The number of label classes
    @param batch_size: Model parameter that can be tuned to alter the speed of model training and its performance
    @param epochs: Model parameter that can be tuned to alter the speed of model training and its performance
    @return: A fitted Keras model
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(n_classes, activation="softmax")
        ]
    )

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs)
    return model

def create_test_eval_dict(test_X):
    """
    Builds a dictionary for evaluating the MNIST test dataset
    @param test_X: The MNIST test dataset
    @return: A dictionary for evaluating the MNIST test dataset
    """
    test_eval = {}
    for n in range(test_X.shape[0]):
        test_eval[n] = {'image': test_X[n], 'scores':[]}
    return test_eval

def batch_run_models(n_runs, train_X, train_y, test_X, test_y, input_shape, n_classes=10, batch_size=128, epochs=3, test_eval_save_file='test_eval_raw.pkl'):
    """
    Builds multiple models, makes predictions on the test dataset, then calculates the log loss function for every
    test sample and saves this to a dictionary.

    @param n_runs: Number of times to run the workflow build, run, predict and evaulate the test dataset
    @param train_X: The MNIST train dataset
    @param train_y: The MNIST train labels
    @param test_X: The MNIST test dataset
    @param test_y: The MNIST test labels
    @param input_shape: The input shape of the train dataset used to build models
    @param n_classes: Number of classes for labels (The MNIST dataset has 10 labels)
    @param batch_size: Model parameter that can be tuned to alter the speed of model training and its performance
    @param epochs: Model parameter that can be tuned to alter the speed of model training and its performance
    @param test_eval_save_file: String path and file name to save MNIST model evaluation data to. Saved as a Pickle file.
    @return: A dictionary representing the evaluation for each MNIST test image.
            dict[index] = dict('image': test data, 'scores': list(log loss scores))
    """
    test_eval = create_test_eval_dict(test_X)

    for n in range(1,n_runs+1):
        model = build_train_model(train_X, train_y, input_shape=input_shape, n_classes=n_classes,
                                        batch_size=batch_size, epochs=epochs)
        pred = model.predict(test_X)

        for i in range(len(pred)):
            loss = log_loss(test_y[i], pred[i])
            test_eval[i]['scores'].append(loss)

        print("Batch runs completed: {} out of {}".format(n, n_runs))

    # Save model evaluations to file
    with open(test_eval_save_file, 'wb') as file:
        pickle.dump(test_eval, file)

    return test_eval

def save_ambiguous_images(test_eval, image_folder_path="image"):
    """
    Evaluates the model predictions on the MNIST test, finds the most ambiguous images and saves them
    to a specified folder.

    First, calculate the mean of all scores and extract the bottom 30 from the dataset as a subset
    Then perform the Student t-test on all the score distributions from this selection of low test evaluation scores
    to find the 10 test datasets which have the 10 lowest true score mean, representing the 10 most ambiguous test datasets

    @param test_eval: Dictionary of evaluation results from MNIST model predictions
    @param image_folder_path: folder path to save the most ambiguous images to
    @return:
    """

    test_eval_list = []
    n_subset = 30

    for v in test_eval.values():
        image = v['image']
        scores = v['scores']
        mean = sum(scores) / len(scores)
        test_eval_list.append([mean, scores, image])

    test_eval_mean_sorted = sorted(test_eval_list, key=lambda x: x[0])

    low_mean_scores = test_eval_mean_sorted[-n_subset:]

    # Score each image to find the lowest scores from paired t-tests
    for image in low_mean_scores:
        image.append(0)

    # Compare each combination of pairs from the subset of low mean evaluation scores.
    # The image with the larger true mean will be given a score of 1.
    # Images with the highest score are the least ambiguous
    # The bottom 10 scores represent the 10 most ambiguous images
    for i in range(n_subset):
        for j in range(n_subset):
            ttest = ttest_ind(low_mean_scores[i][1], low_mean_scores[j][1]).statistic
            if ttest > 0:
                low_mean_scores[i][3] += 1
            elif ttest < 0:
                low_mean_scores[j][3] += 1

    # Sort images by the t-test performance
    ttest_sorted = sorted(low_mean_scores, key=lambda x: x[3])

    ambiguous_images = ttest_sorted[:10]

    # Make a new folder to save images if it doesn't already exist
    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)

    for n, image in enumerate(ambiguous_images):
        im_transform = image[2] * 255
        im_transform = im_transform.reshape(28, 28)
        plt.imshow(im_transform, cmap='gray')
        rank = n + 1
        plt.savefig("{}/ambiguous_{}.png".format(image_folder_path, rank))

    print("Ambiguous images saved to folder: {}".format(image_folder_path))