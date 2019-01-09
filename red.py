
"""""
Author : 홍효정, gywjd8497@naver.com

Supervisor : Na, In Seop, ypencil@hanmail.net

Starting Project : 2019.1.4

"""""

import pandas as pd
import sklearn
import matplotlib.pyplot as plt
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
- 전체 칼럼 중에서, numeric과 non-numeric 을 구분 한다.
"""
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
- numeric column에 대해서 correlation matrix를 구성한 다음, 
- SalePrice에 영향을 미치는 상위 10가지 index에 대해서 correlation matrix를 활용하여 
- sns.heatmap으로 확인함. 
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
plt.savefig('../../assets/images/markdown_img/180606_2223_corr_heatmap_top_10_col.svg')
plt.show()




Numeric Cols
['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SalePrice']
------------
NonNumeric Cols
['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
------------