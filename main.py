import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt


learning_rate = 1e-3   #standard learning rate2
epochs        = 128    #massive data
batch_size    = 64
dropout_rate  = 0.5

train_dataset = tf.data.experimental.load("processed_dataset/train/", (tf.TensorSpec(shape=(300, 300, 3), dtype=tf.float32, name=None),
 tf.TensorSpec(shape=(), dtype=tf.int64, name=None)), compression="GZIP")

valid_dataset = tf.data.experimental.load("processed_dataset/valid/", (tf.TensorSpec(shape=(300, 300, 3), dtype=tf.float32, name=None),
 tf.TensorSpec(shape=(), dtype=tf.int64, name=None)), compression="GZIP")

train_dataset = train_dataset.shuffle(len(train_dataset))
valid_dataset = valid_dataset.shuffle(len(valid_dataset))
train_dataset = train_dataset.batch(batch_size)
valid_dataset = valid_dataset.batch(batch_size)

num_classes = 1

model = Sequential([
 layers.experimental.preprocessing.Rescaling(1./255, input_shape=(300,300,3)),
 layers.experimental.preprocessing.Resizing(100,100),

 layers.Conv2D(32, (3,3), padding = "same", kernel_initializer="he_uniform", activation="relu"),
 layers.MaxPooling2D(2, 2),

 layers.Conv2D(64, (3,3), padding="same", kernel_initializer="he_uniform", activation="relu"),
 layers.MaxPooling2D(2, 2),

 layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_uniform", activation="relu"),
 layers.MaxPooling2D(2, 2),

 layers.Dropout(0.3),

 layers.Flatten(),

 layers.Dense(64, activation="relu", kernel_initializer="he_uniform"),
 layers.Dense(num_classes, activation="sigmoid")

])

MCheck = tf.keras.callbacks.ModelCheckpoint('The_Model.h5', monitor="val_accuracy", verbose=1, save_best_only=True)

opt = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9)

model.compile(optimizer=opt,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
  train_dataset,
  validation_data=valid_dataset,
  epochs=epochs,
  callbacks=MCheck
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()