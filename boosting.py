import xgboost
from sklearn import preprocessing, metrics
import numpy as np
import preproc_data

##### boosted tree
df, labels, pos_weight =  preproc_data.return_data(orig=False)
df = (df - df.mean(axis=0))/df.std(axis=0) ## L-BFGS won't converge to good results without normalization, but SGD is OK without it
p_val = 0.2
rand_perm_ind = np.random.permutation(df.shape[0])  ## split train/val/test
train_ind = rand_perm_ind[:int((1 - 2 * p_val) * len(rand_perm_ind))]
val_ind = rand_perm_ind[int((1 - 2 * p_val) * len(rand_perm_ind)):int((1 - p_val) * len(rand_perm_ind))]
test_ind = rand_perm_ind[int((1 - p_val) * len(rand_perm_ind)):]

train_x = df.iloc[train_ind, :]
train_y = labels[train_ind, :]
val_x = df.iloc[val_ind, :]
val_y = labels[val_ind, :]
test_x = df.iloc[test_ind, :]
test_y = labels[test_ind, :]

xgb = xgboost.XGBClassifier(learning_rate=0.05, max_depth=8, scale_pos_weight=10, n_estimators=100, n_jobs=8, nthread=-1, subsample=.8,
                            colsample_bylevel=.8, colsample_bynode=.8, colsample_bytree=.8, gamma=1, base_score=.5, min_child_weight=1, max_delta_step=0.0) ##n_estimators=1000, learning_rate=0.05
xgb.fit(X=train_x.values.astype(np.float32), y=np.squeeze(train_y.astype(np.float32)), early_stopping_rounds=50,
        eval_set=[(val_x.values.astype(np.float32), val_y.astype(np.float32))], verbose=True)

# make predictions
predxgb = xgb.predict(test_x.values.astype(np.float32))
xgb_conf_mat = metrics.confusion_matrix(test_y.astype(np.float32), predxgb) ## tree methods tend to have higher false negative rates than ANN
print(xgb_conf_mat/np.expand_dims(np.sum(xgb_conf_mat, axis=1), axis=1))

import matplotlib.pyplot as plt
xgboost.plot_importance(xgb)
plt.gca().set_ylim([40,50])