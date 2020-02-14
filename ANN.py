import tensorflow as tf
import preproc_data
import numpy as np
from sklearn import metrics
from earlyStopping import *

##################################### neural network ############################################
def eval_loss_and_grads(x, loss_train, var_list, var_shapes, var_locs):
    ## x: updated variables from scipy optimizer
    ## var_list: list of trainable variables
    ## var_sapes: the shape of each variable to use for updating variables with optimizer values
    ## var_locs:  slicing indecies to use on x prior to reshaping

    ## update variables
    for i in range(len(var_list)):
        var_list[i].assign(np.reshape(x[var_locs[i]:(var_locs[i+1])], var_shapes[i]))

    ## calculate new gradient
    with tf.GradientTape() as tape:
        prediction_loss = loss_train()

    grad_list = []
    for p in tape.gradient(prediction_loss, var_list):
        grad_list.extend(np.array(tf.reshape(p, [-1])))

    grad_list = [v if v is not None else 0 for v in grad_list]

    # return np.float64(prediction_loss), np.float64(grad_list)
    return tf.constant(prediction_loss), tf.constant(grad_list)

def df_to_dataset(dataframe, labels, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    # labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dataframe.values.astype(np.float32), labels.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

def customLoss(yTrue,yPred):
    return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(yTrue, yPred, pos_weight=10))  ## upweight pos error due to 9:1 ratio of neg:pos samples

def customMonitor(yTrue,yPred):
    out = tf.nn.sigmoid(yPred)
    temp = tf.keras.metrics.binary_accuracy(yTrue, out)
    return tf.reduce_mean(temp)

df, labels, pos_weight =  preproc_data.return_data(orig=False)

df = (df - df.mean(axis=0))/df.std(axis=0) ## L-BFGS won't converge to good results without normalization, but SGD is OK without it

split = 0.2

# batch_size = 50000
# train_ds = df_to_dataset(df.iloc[:np.int(df.shape[0]*(1-2*split)), :], labels[:np.int(df.shape[0]*(1-2*split))], batch_size=batch_size)
# val_ds = df_to_dataset(df.iloc[np.int(-2*df.shape[0]*split):np.int(-df.shape[0]*split), :], labels[np.int(-2*df.shape[0]*split):np.int(-df.shape[0]*split)], shuffle=False, batch_size=batch_size)
# test_ds = df_to_dataset(df.iloc[np.int(-df.shape[0]*split):, :], labels[np.int(-df.shape[0]*split):], shuffle=False, batch_size=batch_size)

# actfun = tf.nn.tanh
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(4, activation=actfun, input_dim=df.shape[1]))
# # model.add(tf.keras.layers.Dense(128, activation=actfun))
# # model.add(tf.keras.layers.Dense(32, activation=actfun))
# # model.add(tf.keras.layers.Dense(4, activation=actfun))
# # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# model.add(tf.keras.layers.Dense(1, activation=None)) ## logits

# earl = tf.keras.callbacks.EarlyStopping(monitor='customLoss', min_delta=0, patience=20, verbose=1, mode='auto', baseline=None, restore_best_weights=True) #binary_accuracy  val_loss
# # tens = tf.keras.callbacks.TensorBoard(log_dir=r'C:\Users\justjo\Downloads\bank\ANN/logs', histogram_freq=0, write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
# # # tensorboard --logdir=C:\Users\justjo\Downloads\bank\ANN/logs
#
# # model.compile(optimizer='adam',
# #               loss=customLoss, #'binary_crossentropy'
# #               metrics=[tf.keras.metrics.binary_accuracy])
#
# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss=customLoss, #'binary_crossentropy'
#               metrics=[customLoss, customMonitor]) #accuracy
#
# model.fit(train_ds,
#           validation_data=val_ds,
#           epochs=5000,
#           callbacks=[earl])

feat_np = df.values
targ_np = labels.astype(np.float32)
p_val = 0.2
early_stop = EarlyStopping(model=None, patience=200, monitor=None)  ## set model=model, monitor=loss_val dynamically
for _ in range(5):

    actfun = tf.nn.elu
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4, activation=actfun, input_dim=df.shape[1]))
    # model.add(tf.keras.layers.Dense(128, activation=actfun))
    # model.add(tf.keras.layers.Dense(32, activation=actfun))
    # model.add(tf.keras.layers.Dense(4, activation=actfun))
    # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1, activation=None))  ## logits

    rand_perm_ind = np.random.permutation(feat_np.shape[0])  ## split train/val/test
    early_stop.rand_perm_ind_temp = rand_perm_ind
    train_ind = rand_perm_ind[:int((1 - 2 * p_val) * len(rand_perm_ind))]
    val_ind = rand_perm_ind[int((1 - 2 * p_val) * len(rand_perm_ind)):int((1 - p_val) * len(rand_perm_ind))]
    test_ind = rand_perm_ind[int((1 - p_val) * len(rand_perm_ind)):]

    train_x = feat_np[train_ind, :]
    train_y = targ_np[train_ind, :]
    val_x = feat_np[val_ind, :]
    val_y = targ_np[val_ind, :]
    test_x = feat_np[test_ind, :]
    test_y = targ_np[test_ind, :]

    # with tf.device('/cpu:0'):
    #     model = bayes_model_create(feat_np.shape[1:], actfun, h_units, dist=dist)

    ## optimize
    def value_and_gradients_function(x):
        return eval_loss_and_grads(x, loss_train, var_list, var_shapes, var_locs)

    def loss_train():
        # return tf.reduce_mean(-model(train_x).log_prob(train_y))
        return customLoss(train_y, model(train_x))

    def loss_val():
        # return tf.reduce_mean(-model(val_x).log_prob(val_y)).numpy()
        return customLoss(val_y, model(val_x))


    # def loss_test():
    #     loss_test_ = tf.subtract(test_y,model(test_x, training=False))
    #     return tf.reduce_mean(tf.square(loss_test_) / tf.sqrt(tf.square(loss_test_) + .01 ** 2)).numpy()

    with tf.GradientTape() as tape:
        prediction_loss = loss_train()
        var_list = tape.watched_variables()

    var_shapes = []
    var_locs = [0]
    for v in var_list:
        var_shapes.append(np.array(tf.shape(v)))
        var_locs.append(np.prod(np.array(tf.shape(v))))
    var_locs = np.cumsum(var_locs)

    x_init = []
    for v in var_list:
        x_init.extend(np.array(tf.reshape(v, [-1])))

    start_ = np.array(x_init)

    ## reset early stopping monitor values (except for best weights & best val to beat, which are restored at end)
    early_stop.monitor = loss_val
    early_stop.model = model
    early_stop.wait = 0
    early_stop.stop_training = False

    optimizer_results = tfp.optimizer.lbfgs_minimize(
        value_and_gradients_function,
        start_,
        num_correction_pairs=10,
        tolerance=1e-08,
        x_tolerance=0,
        f_relative_tolerance=0,
        initial_inverse_hessian_estimate=None,
        max_iterations=5000,
        parallel_iterations=1,
        stopping_condition=early_stop.on_epoch_end,
        name=None
    )

out = tf.nn.sigmoid(model(df.values.astype(np.float32)))
binacc=tf.keras.metrics.binary_accuracy(labels, out)
print(np.mean(binacc))

binacc_pos=tf.keras.metrics.binary_accuracy(labels[labels==1], out[labels==1])
print(np.mean(binacc_pos))

# loss, accuracy = model.evaluate(test_ds)
# print("Accuracy", accuracy)

test = [x for x in test_ds]
xgbrf_conf_mat = metrics.confusion_matrix(test[0][1].numpy(), tf.nn.sigmoid(model(test[0][0])).numpy().round())
print(xgbrf_conf_mat/np.expand_dims(np.sum(xgbrf_conf_mat, axis=1), axis=1))