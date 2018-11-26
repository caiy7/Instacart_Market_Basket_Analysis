#!/usr/bin/python

import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, recall_score, precision_score, \
f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve
import xgboost as xgb

with open('../data/final_dataset.pkl', 'rb') as f:
    train_df, test_df = pickle.load(f)

y_train = train_df['reordered']
x_train = train_df.drop(columns=['reordered', 'user_id', 'product_id'])
feature_names = x_train.columns
print('Start training model...')

gbm =  xgb.XGBClassifier( 
                       n_estimators = 843, #arbitrary large number
                       max_depth = 6,
                       objective = "binary:logistic",
                       learning_rate = 0.01, 
                       subsample = 0.3,
                       min_child_weight = 1,
                       colsample_bytree = 0.7
                          )
fit_model = gbm.fit( 
            x_train, y_train, 
           )
with open('../result/xgb_model.pkl', 'wb') as f:
    pickle.dump(fit_model, f)
with open('../result/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print('Model saved.\nMaking prediction')

x_test = test_df.drop(columns=['reordered', 'user_id', 'product_id'])
y_test = test_df['reordered']

y_pred_prob = gbm.predict_proba(x_test, ntree_limit=843)[:,1]
y_pred = y_pred_prob>0.2114
y_pred = y_pred.astype('int')

f1 = f1_score(y_test, y_pred)
print('F1 score is ', f1)
rec = recall_score(y_test, y_pred)
print('Recall is ', rec)
prec = precision_score(y_test, y_pred)
print('Precision is ', prec)

with open('../result/y_pred_prob.pkl', 'wb') as f:
    pickle.dump(y_pred_prob, f)