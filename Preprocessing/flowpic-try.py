#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras import backend as K
from keras.metrics import top_k_categorical_accuracy
import os


# In[3]:


K.set_image_data_format('channels_first')
print(K.backend(), K.image_data_format())


# In[4]:


batch_size = 128 #128
samples_per_epoch = 10
num_classes = 4
epochs = 5 #40
class_names = ["voip", "video", "file transfer", "chat"]

# input hist dimensions
height, width = 1500, 1500
input_shape = (1, height, width)
#input_shape = (height, width)
MODEL_NAME = "FlowPic"


# In[22]:


# Reading the data from the uploaded in Kaggle's input
path = "/kaggle/input/vpn-non-vpn-iscx-2016-npy"
X_train = np.load('/kaggle/input/vpn-non-vpn-iscx-2016-npy/reg_X_train.npy')#, mmap_mode='r')
y_train_true = np.load('/kaggle/input/vpn-non-vpn-iscx-2016-npy/reg_y_train.npy')#, mmap_mode='r')
X_val = np.load('/kaggle/input/vpn-non-vpn-iscx-2016-npy/reg_X_val.npy')#, mmap_mode='r')
y_val_true = np.load('/kaggle/input/vpn-non-vpn-iscx-2016-npy/reg_y_val.npy')#, mmap_mode='r')


# In[23]:


X_train = np.expand_dims(X_train, axis=1)
# X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_val = np.expand_dims(X_val, axis=1)

print(X_train.shape)  # Should print (420, 1, 1500, 1500)


# In[24]:


# X_train.shape, X_val.shape, y_train.shape, y_val.shape
X_train.shape, X_val.shape, y_train_true.shape, y_val_true.shape


# In[26]:


# # Transpose the data from (1, 1500, 1500) to (1500, 1500, 1)
# X_train = np.transpose(X_train, (0, 2, 3, 1))  # (samples, channels, height, width) -> (samples, height, width, channels)
# X_val = np.transpose(X_val, (0, 2, 3, 1))

# print(X_train.shape)  # Should print (num_samples, 1500, 1500, 1)
# print(X_val.shape)


# In[9]:


print(y_train_true[0:70])


# In[10]:


y_train = to_categorical(y_train_true, num_classes)
y_val = to_categorical(y_val_true, num_classes)
print(y_train[0:10])
print (y_val[0:10])
print(y_train.shape, y_val.shape)


# In[27]:


# def precision(y_true, y_pred):
#     """Precision metric.

#     Only computes a batch-wise average of precision.

#     Computes the precision, a metric for multi-label classification of
#     how many selected items are relevant.
#     """
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def recall(y_true, y_pred):
#     """Recall metric.

#     Only computes a batch-wise average of recall.

#     Computes the recall, a metric for multi-label classification of
#     how many relevant items are selected.
#     """
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def f1_score(y_true, y_pred):
#     prec = precision(y_true, y_pred)
#     rec = recall(y_true, y_pred)
#     return 2*((prec*rec)/(prec+rec))

import tensorflow.keras.backend as K

# Define custom metrics (if required)
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))
def top_2_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2) 


# In[32]:


from keras.layers import Activation

# model = Sequential()
# # model.add(BatchNormalization(input_shape=input_shape, axis=-1, momentum=0.99, epsilon=0.001)) ############################
# model.add(Conv2D(10, kernel_size=(10, 10),strides=5,padding="same", input_shape=input_shape))
# convout1 = Activation('relu')
# model.add(convout1)
# print(model.output_shape)
# model.add(MaxPooling2D(pool_size=(2, 2)))
# print(model.output_shape)
# model.add(Conv2D(20, (10, 10),strides=5,padding="same"))  #################################################
# convout2 = Activation('relu')
# model.add(convout2)
# print(model.output_shape)
# model.add(Dropout(0.25))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# print(model.output_shape)
# model.add(Flatten())
# print(model.output_shape)
# model.add(Dense(64, activation='relu'))
# print(model.output_shape)
# model.add(Dropout(0.5))
# print(model.output_shape)
# model.add(Dense(num_classes, activation='softmax'))
# print(model.output_shape)

# # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', top_2_categorical_accuracy, f1_score, precision, recall])
# # Compile the model
# # model.compile(
# #     loss='categorical_crossentropy',
# #     optimizer='adam',
# #     metrics=['accuracy', top_2_categorical_accuracy, precision, recall, f1_score]
# # )

# Input data is already in (batch_size, channels, height, width)
print(X_train.shape)  # Should be (num_samples, 1, 1500, 1500)

#Model definition

model = Sequential()
model.add(Conv2D(10, kernel_size=(10, 10), strides=5, padding="same", data_format='channels_last', input_shape=(1500, 1500,1)))
print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2, 2), padding="same", data_format='channels_last'))
print(model.output_shape)
model.add(Conv2D(20, kernel_size=(10, 10), strides=5, padding="same", data_format='channels_last'))
print(model.output_shape)
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same", data_format='channels_last'))
print(model.output_shape)
model.add(Flatten())
print(model.output_shape)
model.add(Dense(64, activation='relu'))
print(model.output_shape)
model.add(Dropout(0.5))
print(model.output_shape)
model.add(Dense(num_classes, activation='softmax'))
print(model.output_shape)

from tensorflow.keras.metrics import Precision, Recall

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy',top_2_categorical_accuracy ,Precision(), Recall()]
)

# Define the model
# model = Sequential()
# model.add(Conv2D(10, kernel_size=(10, 10), strides=5, padding="same", input_shape=(1500, 1500, 1)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
# model.add(Conv2D(20, kernel_size=(10, 10), strides=5, padding="same"))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
# model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# # Compile the model
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy', Precision(), Recall()]
# )


# In[33]:


from tensorflow.keras import backend as K

print(K.image_data_format())  # Should print "channels_last"
if K.image_data_format() != "channels_last":
    K.set_image_data_format("channels_last")


# In[34]:


# from tensorflow.keras import backend as K
# K.set_image_data_format('channels_last')
# print(K.image_data_format())


# In[35]:


from keras.callbacks import TensorBoard, ModelCheckpoint

# Define TensorBoard
tensorboard = TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)

# Define ModelCheckpoint
# checkpointer_loss = ModelCheckpoint(filepath=MODEL_NAME + '_loss.hdf5', verbose=1, save_best_only=True, save_weights_only=True)
# checkpointer_acc = ModelCheckpoint(monitor='val_acc', filepath=MODEL_NAME + '_acc.hdf5', verbose=1, save_best_only=True, save_weights_only=True)

#Define generator
def generator(features, labels, batch_size):
    index = 0
    while True:
        index += batch_size
        if index >= len(features):
            batch_features = np.append(features[index - batch_size:len(features)], features[0:index - len(features)], axis=0)
            batch_labels = np.append(labels[index - batch_size:len(features)], labels[0:index - len(features)], axis=0)
            index -= len(features)
            yield batch_features, batch_labels
        else:
            yield features[index - batch_size:index], labels[index - batch_size:index]

# def generator(features, labels, batch_size):
#     index = 0
#     while True:
#         if index + batch_size > len(features):
#             batch_features = np.concatenate((features[index:], features[:(index + batch_size) % len(features)]), axis=0)
#             batch_labels = np.concatenate((labels[index:], labels[:(index + batch_size) % len(labels)]), axis=0)
#         else:
#             batch_features = features[index:index + batch_size]
#             batch_labels = labels[index:index + batch_size]
#         index = (index + batch_size) % len(features)
#         yield batch_features, batch_labels

# Train the model

history = model.fit(
    generator(X_train, y_train, batch_size),
    epochs=epochs,
    steps_per_epoch=samples_per_epoch, #// batch_size,
    verbose=1,
    callbacks=[tensorboard], #, checkpointer_loss, checkpointer_acc],
    validation_data=(X_val, y_val)
)


# In[37]:


import matplotlib.pyplot as plt
# List all available keys in history
print(history.history.keys())

# Define x-axis values (epochs)
x = np.asarray(range(1, epochs + 1))

# Plot accuracy
plt.figure()
plt.plot(x, history.history['accuracy'], label='Train Accuracy')  # Use 'accuracy' instead of 'acc'
plt.plot(x, history.history['val_accuracy'], label='Validation Accuracy')  # Use 'val_accuracy' instead of 'val_acc'
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()


# In[39]:


# Get predictions (probabilities for each class)
y_val_probabilities = model.predict(X_val, verbose=1)

# Convert probabilities to class predictions
y_val_predictions = np.argmax(y_val_probabilities, axis=1)

print(y_val_predictions)


# In[41]:


from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          fname='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, format(cm[i, j]*100, fmt) + '%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")    
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(fname, bbox_inches='tight', pad_inches=1)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_val_true, y_val_predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization',
                      fname=MODEL_NAME + "_" + 'Confusion_matrix_without_normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix',
                      fname=MODEL_NAME + "_" + 'Normalized_confusion_matrix')

plt.show()


# In[42]:


y_val_true = np.argmax(y_val, axis=1)

# Compare predictions with true labels
accuracy = np.mean(y_val_predictions == y_val_true)
print(f"Validation Accuracy: {accuracy:.2f}")


# In[43]:


from sklearn.metrics import classification_report, accuracy_score

# Print classification report
print(classification_report(y_val_true, y_val_predictions))

# Calculate and print accuracy as a percentage
accuracy = accuracy_score(y_val_true, y_val_predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[ ]:




