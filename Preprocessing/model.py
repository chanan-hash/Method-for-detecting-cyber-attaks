from tensorflow.keras import models, layers

model = models.Sequential()
# Args: filters, kernel_size, strides
model.add(layers.Conv2D(10, (10,10), (5,5), activation='relu', input_shape=INPUT_SHAPE))
# Args: pool_size, strides (defaults to pool)
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(20, (10,10), (5,5), activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
# Binary classification, turn into probabilities
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()