import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
# additional packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import time
from sklearn.metrics import f1_score
# importing random for randint() function
import random
# importing keras to use Sequential class
import keras
from keras.models import Sequential
from tensorflow import keras
# importing for visualization graphs
from sklearn.metrics import precision_recall_curve, roc_curve, auc
# importing f1_score to assess precision
from sklearn.metrics import f1_score

df = pd.read_csv('dataset.csv')

y = df['Positive Review']
X = df['Review']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# creation of TF-IDF vectorizer object, fitting to training data
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(X_train)

# transform the training data using fitted vectorizer
X_train_tfidf = tfidf_vectorizer.transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# finding dimensionality of input layer
vocabulary_size = len(tfidf_vectorizer.vocabulary_)
print(vocabulary_size)

# creating neural network 
nn_model = keras.Sequential()

# input layer, 3 hidden layers, output layer
input_layer = keras.layers.InputLayer(input_shape=(vocabulary_size,))
nn_model.add(input_layer)

hidden_layer_1 = keras.layers.Dense(units=64, activation='relu')
nn_model.add(hidden_layer_1)

hidden_layer_2 = keras.layers.Dense(units=32, activation='relu')
nn_model.add(hidden_layer_2)

hidden_layer_3 = keras.layers.Dense(units=16, activation='relu')
nn_model.add(hidden_layer_3)

output_layer = keras.layers.Dense(units=1, activation='sigmoid')
nn_model.add(output_layer)

# output neural network model structure
nn_model.summary()

# optimization function & loss function
sgd_optimizer = keras.optimizers.SGD(learning_rate=0.1)
loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)

# compiling model
nn_model.compile(optimizer=sgd_optimizer, loss=loss_fn, metrics=['accuracy'])

# fit model on training data
class ProgBarLoggerNEpochs(keras.callbacks.Callback):
    
    def __init__(self, num_epochs: int, every_n: int = 50):
        self.num_epochs = num_epochs
        self.every_n = every_n
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n == 0:
            s = 'Epoch [{}/ {}]'.format(epoch + 1, self.num_epochs)
            logs_s = ['{}: {:.4f}'.format(k.capitalize(), v)
                      for k, v in logs.items()]
            s_list = [s] + logs_s
            print(', '.join(s_list))

# initialize start time and epochs     
t0 = time.time() 
num_epochs = 50

# converting to NumPy array
X_train_tfidf_np = X_train_tfidf.toarray()

# fitting model
history = nn_model.fit(X_train_tfidf_np, y_train, epochs=num_epochs, verbose=0, callbacks=[ProgBarLoggerNEpochs(num_epochs, every_n=50)], validation_split=0.2)
t1 = time.time()
print('Elapsed time: %.2fs' % (t1-t0))

# visualization for loss over time
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), history.history['loss'], label='Training Loss')
plt.plot(range(1, num_epochs + 1), history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()

# visualization for accuracy over time
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), history.history['accuracy'], label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Time')
plt.legend()
plt.grid(True)
plt.show()

# evaluate performance on test data
X_test_tfidf_np = X_test_tfidf.toarray()
loss, accuracy = nn_model.evaluate(X_test_tfidf_np, y_test)

print('Loss: ', str(loss) , 'Accuracy: ', str(accuracy))

# predictions on the test set
probability_predictions = nn_model.predict(X_test_tfidf.toarray())

print("Predictions for the first 10 examples:\n")
print("\tProbability\t\tClass")
print("-------------------------------------------------")
for i in range(0,10):
    if probability_predictions[i] >= .5:
        class_pred = "|\tGood Review"
    else:
        class_pred = "|\tBad Review"
    print("\t" + str(probability_predictions[i]) + "\t" + str(class_pred))

# using random class to generate random review from the predictions to be shown
print('Test #1:\n')
# probability_predictions has a limit of 395 so the random number generator only reaches 395
num1 = random.randint(0,395)
print(X_test.to_numpy()[num1])
goodReview = True if probability_predictions[num1] >= .5 else False
print('\nPrediction: Is this a good review? {}\n'.format(goodReview))
print('Actual: Is this a good review? {}\n'.format(y_test.to_numpy()[num1]))

print('Test #2:\n')
num1 = random.randint(0,395)
print(X_test.to_numpy()[47])
goodReview = True if probability_predictions[47] >= .5 else False
print('\nPrediction: Is this a good review? {}\n'.format(goodReview)) 
print('Actual: Is this a good review? {}\n'.format(y_test.to_numpy()[47]))

# calculate the predicted labels based on the threshold
predicted_labels = [1 if prob >= 0.5 else 0 for prob in probability_predictions]

# f1 score
f1 = f1_score(y_test, predicted_labels)
print("F1 Score:", f1)

''' A higher F1 score is preferred. F1 scores are from 0.0 to 1.0. 
A higher F1 score means that the model is performing well at finding the right 
balance between precision and recall. It suggests that the model is correctly classifying a
significant portion of both positive and negative instances (in this model, it would be both 
Good Reviews and Bad Reviews) in the test dataset. ''' 

# visualization for precision-recall curve
# best values for precision-recall balance in the model are closest to the top right corner
precision, recall, _ = precision_recall_curve(y_test, probability_predictions)
plt.plot(recall, precision, color='b', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# visualization for ROC curve
# higher area under curve (AUC) demonstrates better discrimination abilities
fpr, tpr, _ = roc_curve(y_test, probability_predictions)
roc_auc = auc(fpr, tpr)
# showing AUC value
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()



