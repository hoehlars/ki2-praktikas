"""
Hate speech classification baseline using sklearn
Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
"""

__author__ = "don.tuggener@zhaw.ch"

import csv
import pdb
import re
import pdb
import sys
import pickle
import random
import zipfile

from collections import Counter
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np

import tensorflow as tf
from tensorflow import keras
import spacy
import matplotlib.pyplot as plt

random.seed(42)  # Ensure reproducible results
STEMMER = SnowballStemmer("english")
STOPWORDS = stopwords.words('english')


def read_data(remove_stopwords=True, remove_numbers=True, do_stem=True, reprocess=False):
    """ 
    Read CSV with annotated data. 
    We'll binarize the classification, i.e. subsume all hate speach related classes 
    'toxic, severe_toxic, obscene, threat, insult, identity_hate'
    into one.

    In this method we also do a lot of preprocessing steps, based on the flags which are set in the parameters.
    Feel free to try out different possible combinations of preprocessing steps (e.g. with cross-validation).
    """
    if reprocess:
        X, Y = [], []
        zip_ref = zipfile.ZipFile('train.csv.zip', 'r')
        zip_ref.extractall()
        zip_ref.close()
        for i, row in enumerate(csv.reader(open('train.csv', encoding='UTF-8'))):
            if i > 0:   # Skip the header line
                sys.stderr.write('\r'+str(i))
                sys.stderr.flush()
                text = re.findall('\w+', row[1].lower())
                if remove_stopwords:
                    text = [w for w in text if not w in STOPWORDS]
                if remove_numbers:
                    text = [w for w in text if not re.sub('\'\.,','',w).isdigit()]
                if do_stem:
                    text = [STEMMER.stem(w) for w in text]
                label = 1 if '1' in row[2:] else 0  # Any hate speach label 
                X.append(' '.join(text))
                Y.append(label)
        sys.stderr.write('\n')
        pickle.dump(X, open('X.pkl', 'wb'))
        pickle.dump(Y, open('Y.pkl', 'wb'))
    else:
        X = pickle.load(open('X.pkl', 'rb'))
        Y = pickle.load(open('Y.pkl', 'rb'))
    print(len(X), 'data points read')
    print('Label distribution:',Counter(Y))
    print('As percentages:')
    for label, count_ in Counter(Y).items():
        print(label, ':', round(100*(count_/len(X)), 2))
    return X, Y


if __name__ == '__main__':

    print('Loading data', file=sys.stderr)
    X, Y = read_data(reprocess=False)

    print('Vectorizing with TFIDF', file=sys.stderr)
    tfidfizer = TfidfVectorizer(max_features=1000)
    X_tfidf_matrix = tfidfizer.fit_transform(X)
    print('Data shape:', X_tfidf_matrix.shape)
    do_downsample = True
    if do_downsample:   # Only take 20% of the data
        X_tfidf_matrix, X_, Y, Y_ = train_test_split(X_tfidf_matrix, Y, test_size=0.8, random_state=42, stratify=Y)
        print('Downsampled data shape:', X_tfidf_matrix.shape)

    print('Classification and evaluation', file=sys.stderr)
    #clf = LinearSVC(class_weight='balanced')    # Weight samples inverse to class imbalance
    # Randomly split data into 80% training and 20% testing, preserve class distribution with stratify
    X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf_matrix, Y, test_size=0.2, random_state=42, stratify=Y)

    #clf.fit(X_train, Y_train)
    #y_pred = clf.predict(X_test)
    #print(classification_report(Y_test, y_pred), file=sys.stderr)
    #print(confusion_matrix(Y_test, y_pred.tolist()), file=sys.stderr)


    use_dnn = True
        if use_dnn:
        """
        # Apply cross-validation, create prediction for all data point
        numcv = 3   # Number of folds
        print('Using', numcv, 'folds', file=sys.stderr)
        y_pred = cross_val_predict(clf, X_tfidf_matrix, Y, cv=numcv)
        print(classification_report(Y, y_pred), file=sys.stderr)
        """
        
        """
        Train the model 
        """
        
        model = keras.Sequential()
        model.add(keras.layers.Dense(500, input_shape=(X_tfidf_matrix.shape[1],)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(250, input_shape=(500,)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(100, input_shape=(250,)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(1, input_shape=(100,)))
        model.add(keras.layers.Activation('sigmoid'))
        # last result: loss: 0.0352 - accuracy: 0.9586 - val_loss: 0.0391 - val_accuracy: 0.9547
        
        opt = keras.optimizers.SGD(lr=0.1) #Default lr=0.01
        
        model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
        print(model.summary())
        history = model.fit(X_train.toarray(), np.array(Y_train), epochs=30, batch_size=64, validation_data=(X_test.toarray(), np.array(Y_test)))
        print(history.history)
        
    use_cnn = False
    if use_cnn:
        model = keras.Sequential()
        model.add(keras.layers.Conv1D(128, kernel_size=4, activation='relu'))
        model.add(keras.layers.MaxPooling1D(pool_size=3))
        model.add(keras.layers.LSTM(32, recurrent_dropout = 0.4))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(1, activation = "sigmoid"))
        
        model.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit(X_train, Y_train, epochs=30, batch_size=64, validation_data=(X_test, Y_test))
        print(model.summary())
        print(history.history)    
    

    """
    Plot different statistics about the model
    """
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    accuracy = history_dict['accuracy']
    val_accuracy = history_dict['val_accuracy']
     
    epochs = range(1, len(loss_values) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    #
    # Plot the model accuracy vs Epochs
    #
    ax[0].plot(epochs, accuracy, 'bo', label='Training accuracy')
    ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    ax[0].set_title('Training & Validation Accuracy', fontsize=16)
    ax[0].set_xlabel('Epochs', fontsize=16)
    ax[0].set_ylabel('Accuracy', fontsize=16)
    ax[0].legend()
    #
    # Plot the loss vs Epochs
    #
    ax[1].plot(epochs, loss_values, 'bo', label='Training loss')
    ax[1].plot(epochs, val_loss_values, 'b', label='Validation loss')
    ax[1].set_title('Training & Validation Loss', fontsize=16)
    ax[1].set_xlabel('Epochs', fontsize=16)
    ax[1].set_ylabel('Loss', fontsize=16)
    ax[1].legend()
    
    y_pred = model.predict(X_test.toarray(), batch_size=64, verbose=1)
    y_pred_bool = np.around(y_pred)
    
    print(np.array(Y_test))
    print(y_pred_bool)
    print(classification_report(np.array(Y_test), y_pred_bool))