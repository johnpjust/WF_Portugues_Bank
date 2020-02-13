import xgboost
from sklearn import preprocessing, metrics

#### random forest
xgbrf = xgboost.XGBRFClassifier(max_depth=8, scale_pos_weight=9, n_estimators=100)
# xgbrf = xgboost.XGBRFClassifier(scale_pos_weight=9)
xgbrf.fit(X=train.values.astype(np.float32), y=np.squeeze(train_labels.astype(np.float32)), early_stopping_rounds=20,
        eval_set=[(val.values.astype(np.float32), val_labels.astype(np.float32))], verbose=True)
predxgbrf = xgbrf.predict(test.values.astype(np.float32))
xgbrf_conf_mat = metrics.confusion_matrix(test_labels.astype(np.float32), predxgbrf) ## tree methods tend to have higher false negative rates than ANN
print(xgbrf_conf_mat/np.expand_dims(np.sum(xgbrf_conf_mat, axis=1), axis=1))