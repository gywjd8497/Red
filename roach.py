
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import skew, pearsonr

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb

"""
- 전체
칼럼
중에서, numeric과
non - numeric
을
구분합니다.
"""
train_df = pd.read_csv('./input/train.csv')
cols = train_df.columns
NumericCols = []
nonNumericCols = []
for c in cols:
    col_dtype= train_df[c].dtype
    if col_dtype in ['int64', 'float64']:
        NumericCols.append(c)
    else:
        nonNumericCols.append(c)
print("Numeric Cols")
print(NumericCols)
print("------------")
print("NonNumeric Cols")
print(nonNumericCols)
print("------------")

"""
- numeric
column에
대해서
correlation
matrix를
구성한
다음,
- SalePrice에
영향을
미치는
상위
10
가지
index에
대해서
correlation
matrix를
활용하여
- sns.heatmap으로
확인함.
"""
top_10_index = train_df[NumericCols].corr()['SalePrice'].sort_values(ascending=False)[:10].index

plt.figure(figsize=(15, 6))
sns.heatmap(train_df[top_10_index].corr(), 
            annot=True, 
            linewidths = 3, 
            cbar=True, 
            fmt=".2f"
           )

plt.tick_params(labelsize=13)
plt.gca().xaxis.tick_top() 
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

"""
train_df, test_df
모두에
같은
칼럼
set를
가진
칼럼
선정
"""

test_df = pd.read_csv('./input/test.csv')
both_non_numeric_cols = []
for col in nonNumericCols:
    try:
        if set(train_df[col])==set(test_df[col]):
            both_non_numeric_cols.append(col)
    except:
        continue
print(both_non_numeric_cols)

"""
preprocessing and modeling
"""
input_df = pd.DataFrame({
        'OverallQual': input_df['OverallQual'], # categorical 
        'GrLivArea': np.log(input_df['GrLivArea']), #categorical, skewed
        'GarageCars': input_df['GarageCars'].fillna(input_df['GarageCars'].mean()),# test case, nan exists
        'GarageArea': input_df['GarageArea'].fillna(input_df['GarageArea'].mean()), # test case, nan exists
        'TotalBsmtSF': input_df['TotalBsmtSF'].fillna(input_df['TotalBsmtSF'].mean()), # test case, nan exists
        'FullBath': input_df['FullBath'], 
        'YearBuilt': input_df['YearBuilt'],
        'YearRemodAdd': input_df['YearRemodAdd'],
        'GarageYrBlt': input_df['GarageYrBlt'].fillna(input_df['GarageYrBlt'].mean()),
        'TotalSF': input_df['TotalBsmtSF'].fillna(input_df['TotalBsmtSF'].mean()) + input_df['1stFlrSF'].fillna(input_df['1stFlrSF'].mean()) + input_df['2ndFlrSF'].fillna(input_df['2ndFlrSF'].mean()),

        #'MasVnrArea':input_df['MasVnrArea'].fillna(input_df['MasVnrArea'].mean()),
        #'Fireplaces':input_df['Fireplaces'].fillna(input_df['Fireplaces'].mean()),
        #'TotRmsAbvGrd':input_df['TotRmsAbvGrd'].fillna(input_df['TotRmsAbvGrd'].mean()),   
    })
""""
아래
변수들을
추가했는데(이
변수들은
SalePrice와의
corr이
0.3
정도)
오히려, r2_score가
떨어져서
제외함.
"""
for col in both_non_numeric_cols:
    r = r.join(pd.get_dummies(input_df[col].fillna('None'), prefix=col))
r['HasBsmt'] = r['TotalBsmtSF'] > 0# TotalBsmtSF가 0인 경우는 basement가 없는 경우이므로, feature를 새로 만들어준다.
r['TotalSF'] = np.log1p(r['TotalSF']) # skewness를 조절
r['TotalBsmtSF'] = np.log1p(r['TotalBsmtSF']) # skewness를 조절
"""
최종
결정된
column들의
missing
value, skewness
등을
체크한다.
"""
def print_missing_count_and_skewness():
    temp_r =pd.DataFrame({'missing_v':[r[col].isnull().sum() for col in r.columns],
                         'skewness':[skew(r[col]) for col in r.columns],
                         'uniq_count':[len(set(r[col])) for col in r.columns]
                        }, index=r.columns)
    return temp_r[temp_r['uniq_count']>5]
#print(print_missing_count_and_skewness())

r = pd.DataFrame(MinMaxScaler().fit_transform(r), columns = r.columns)
r = pd.DataFrame(RobustScaler().fit_transform(r), columns = r.columns)

x_train = preprocessingX(train_df)
x_test = preprocessingX(test_df)

y_true = train_df['SalePrice']
y_true_log = np.log(train_df['SalePrice'])
"""

"""
models = [RandomForestRegressor(n_estimators=n, random_state=42) for n in [10, 30, 50, 100]]
models+=[Lasso(alpha =0.0005, random_state=1)]
models+=[ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)]
"""
KernelRidge를
gridsearch로
돌려본
결과, degree가
3
일
때
더
좋았음.
"""
models+=[KernelRidge(alpha=0.6, kernel='polynomial', degree=i, coef0=2.5) for i in range(2, 10)]

models+=[GradientBoostingRegressor(n_estimators=n_e, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5) for n_e in [3000, 5000]]
models+=[xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)]
models+=[xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=5, 
                             min_child_weight=1.7817, n_estimators=5000,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)]
models+=[xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=10, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)]


x_train_sub1, x_train_sub2, y_train_sub1, y_train_sub2 = train_test_split(x_train, y_true_log, train_size=0.7, 
                                                                          random_state=42)

print("---")
for i,m in enumerate(models):
    print(i, m.__class__)
    m.fit(x_train_sub1, y_train_sub1)
    print("sub1: {}, sub2: {}".format(
        r2_score(y_train_sub1, m.predict(x_train_sub1)), 
        r2_score(y_train_sub2, m.predict(x_train_sub2))
    ))
    print("----")

# r2_score가 높은 순으로 model을 정렬해준다. 
models = sorted(models, key=lambda m: r2_score(y_train_sub2, m.predict(x_train_sub2)), reverse=True)

y_preds = np.array([m.predict(x_train_sub2) for m in models]).T
#y_preds_mean = y_preds.mean(axis=1)
y_preds_mean = y_preds.dot(np.linspace(1.0, 0.0, len(models))/sum(np.linspace(1.0, 0.0, len(models))))
print("train test set, r2_score: {}".format(r2_score(np.exp(y_train_sub2), np.exp(y_preds_mean))))
"""
- test_score에
대해서
r2_score가
높은
대로
가중치를
주고
곱하여, y_pred_log를
계산한다.
"""
y_pred_log = np.array([m.predict(x_test) for m in models]).T.dot(
    np.linspace(1.0, 0.0, len(models))/sum(np.linspace(1.0, 0.0, len(models))))

submit_df = pd.DataFrame({'Id':test_df['Id'], 'SalePrice':np.exp(y_pred_log)})
submit_df.to_csv('kaggle_house_price.csv', index=False)
print('complete')