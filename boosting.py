import xgboost
from sklearn import preprocessing, metrics

##### boosted tree
# xgb = xgboost.XGBClassifier(max_depth=8, scale_pos_weight=9, n_estimators=1000) ##n_estimators=1000, learning_rate=0.05
# xgb.fit(X=train.values.astype(np.float32), y=np.squeeze(train_labels.astype(np.float32)), early_stopping_rounds=20,
#         eval_set=[(val.values.astype(np.float32), val_labels.astype(np.float32))], verbose=True)
#
# # make predictions
# predxgb = xgb.predict(test.values.astype(np.float32))
# xgb_conf_mat = metrics.confusion_matrix(test_labels.astype(np.float32), predxgb) ## tree methods tend to have higher false negative rates than ANN
# print(xgb_conf_mat/np.expand_dims(np.sum(xgb_conf_mat, axis=1), axis=1))

# xgb.feature_importances_
