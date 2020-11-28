# Call functions
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# load dataset
DE = pd.read_excel('hv_des_new.xlsx')
array = DE.values
X = array[:,4:]
Y = array[:,0]


# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9, test_size=0.1,random_state=100, shuffle=True)


#scale the data

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



# XGB model construction
xgb_model = xgb.XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=1000, verbosity=1, objective='reg:squarederror',
                             booster='gbtree', tree_method='auto', n_jobs=1, gamma=0.0001, min_child_weight=8,max_delta_step=0,
                             subsample=0.6, colsample_bytree=0.7, colsample_bynode=1, reg_alpha=0,
                             reg_lambda=4, scale_pos_weight=1, base_score=0.6, missing=None,
                             num_parallel_tree=1, importance_type='gain', eval_metric='rmse',nthread=4).fit(X_train,Y_train)

X_pred =xgb_model.predict(X_test)
mse = mean_absolute_error(Y_test, X_pred)
print("MeanAbsoluteError: " + str(mse))

# Prediction
prediction = pd.read_excel('pred_hv_descriptors_new.xlsx')
a = prediction.values
b = a[:,4:]
result=xgb_model.predict(b)
composition=pd.read_excel('pred_hv_descriptors_new.xlsx',sheet_name='Sheet1', usecols="A")
composition=pd.DataFrame(composition)
result=pd.DataFrame(result)
predicted=np.column_stack((composition,result))
predicted=pd.DataFrame(predicted)
predicted.to_excel('predicted_hv_new.xlsx', index=False, header=("RPM","Predicted Vickers hardness"))
print("A file named predicted_hv_new.xlsx has been generated.\nPlease check your folder.")