from __future__ import division
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle

print('Load data...')
df_train = pd.read_csv('./data/train.txt', sep=' ')
df_test = pd.read_csv('./data/test.txt', sep=' ')

df_train['date_numeric'] = pd.to_datetime(df_train['date']).astype(int) / 10**9 / 86400
df_test['date_numeric'] = pd.to_datetime(df_test['date']).astype(int) / 10**9 / 86400

label_encoder = LabelEncoder()
df_train['type_encoded'] = label_encoder.fit_transform(df_train['type'])
df_test['type_encoded'] = label_encoder.transform(df_test['type'])

with open('type_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

y_train = df_train['revenue']
y_test = df_test['revenue']
X_train = df_train.drop(['revenue', 'date', 'type'], axis=1)
X_test = df_test.drop(['revenue', 'date', 'type'], axis=1)

lgb_train = lgb.Dataset(X_train, y_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 63,
	'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

num_leaf = 63

print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_train)

print('Save model...')
gbm.save_model('model.txt')

print('Calculate feature importances...')
print('Feature importances:', list(gbm.feature_importance()))
print('Feature importances:', list(gbm.feature_importance("gain")))

print('\nEvaluating on test set...')
y_pred_test = gbm.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f'\nModel Performance:')
print(f'Mean Squared Error (MSE): ${mse:.2f}')
print(f'Root Mean Squared Error (RMSE): ${rmse:.2f}')
print(f'Mean Absolute Error (MAE): ${mae:.2f}')
print(f'R² Score: {r2:.4f}')

print('\nSample Predictions (first 10):')
print(f'{"Actual Revenue":<20} {"Predicted Revenue":<20} {"Difference":<15} {"Diff %":<10}')
print('-' * 70)
for i in range(min(10, len(y_test))):
    diff = y_pred_test[i] - y_test.iloc[i]
    if abs(y_test.iloc[i]) > 0.01:
        diff_pct = (diff / abs(y_test.iloc[i])) * 100
    else:
        diff_pct = 0.0
    print(f'${y_test.iloc[i]:<18.2f} ${y_pred_test[i]:<18.2f} ${diff:<13.2f} {diff_pct:>7.2f}%')

print('\nTraining complete! Model saved to model.txt')
