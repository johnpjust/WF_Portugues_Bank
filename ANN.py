import tensorflow as tf


##################################### neural network ############################################
def df_to_dataset(dataframe, labels, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    # labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dataframe.values.astype(np.float32), labels.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 50000
train_ds = df_to_dataset(train, train_labels, batch_size=batch_size)
val_ds = df_to_dataset(val, val_labels, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, test_labels, shuffle=False, batch_size=batch_size)

actfun = tf.nn.tanh
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, activation=actfun, input_dim=train.shape[1]))
# model.add(tf.keras.layers.Dense(128, activation=actfun))
# model.add(tf.keras.layers.Dense(32, activation=actfun))
# model.add(tf.keras.layers.Dense(4, activation=actfun))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation=None)) ## logits

earl = tf.keras.callbacks.EarlyStopping(monitor='customLoss', min_delta=0, patience=20, verbose=1, mode='auto', baseline=None, restore_best_weights=True) #binary_accuracy  val_loss
# tens = tf.keras.callbacks.TensorBoard(log_dir=r'C:\Users\justjo\Downloads\bank\ANN/logs', histogram_freq=0, write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
# # tensorboard --logdir=C:\Users\justjo\Downloads\bank\ANN/logs

def customLoss(yTrue,yPred, pos_weight=10):
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(yTrue, yPred, pos_weight=pos_weight))  ## upweight pos error due to 9:1 ratio of neg:pos samples

def customMonitor(yTrue,yPred):
    out = tf.nn.sigmoid(yPred)
    temp = tf.keras.metrics.binary_accuracy(yTrue, out)
    return tf.reduce_mean(temp)

# model.compile(optimizer='adam',
#               loss=customLoss, #'binary_crossentropy'
#               metrics=[tf.keras.metrics.binary_accuracy])

model.compile(optimizer='adam',
              loss=customLoss, #'binary_crossentropy'
              metrics=[customLoss, customMonitor]) #accuracy

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5000,
          callbacks=[earl])

out = tf.nn.sigmoid(model(df.values.astype(np.float32)))
binacc=tf.keras.metrics.binary_accuracy(labels, out)
print(np.mean(binacc))

binacc_pos=tf.keras.metrics.binary_accuracy(labels[labels==1], out[labels==1])
print(np.mean(binacc_pos))

# loss, accuracy = model.evaluate(test_ds)
# print("Accuracy", accuracy)