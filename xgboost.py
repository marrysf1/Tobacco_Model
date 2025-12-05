# 导入工具库
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 用pandas读入数据
data = pd.read_csv('./data/Pima-Indians-Diabetes.csv')

# 做数据切分
train, test = train_test_split(data)

# 特征列
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
# 标签列
target_column = 'Outcome'

# 初始化模型
xgb_classifier = xgb.XGBClassifier(n_estimators=20,\
                                   max_depth=4, \
                                   learning_rate=0.1, \
                                   subsample=0.7, \
                                   colsample_bytree=0.7, \
                                   eval_metric='error')

# Dataframe格式数据拟合模型
xgb_classifier.fit(train[feature_columns], train[target_column])

# 使用模型预测
preds = xgb_classifier.predict(test[feature_columns])

# 判断准确率
print('错误类为%f' %((preds!=test[target_column]).sum()/float(test_y.shape[0])))

