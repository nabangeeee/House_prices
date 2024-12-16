## 1. 데이터 불러오기 & 확인
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')

print(train.head())

print(train.shape)
print(test.shape)


## 2. EDA
## 2-1. 변수 분류
numerical_feats = train.dtypes[train.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))
categorical_feats = train.dtypes[train.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))

print(train[numerical_feats].columns)
print(train[categorical_feats].columns)


## 2-2. 이상치 제거
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
# plt.show()
fig, ax = plt.subplots()
ax.scatter(x = train['GarageArea'], y = train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GarageArea')
# plt.show()
fig, ax = plt.subplots()
ax.scatter(x = train['1stFlrSF'], y = train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('1stFlrSF')
# plt.show()
fig, ax = plt.subplots()
ax.scatter(x = train['2ndFlrSF'], y = train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('2ndFlrSF')
# plt.show()
fig, ax = plt.subplots()
ax.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('TotalBsmtSF')
# plt.show()

train = train[train['GrLivArea']<4000]
train = train[train['GarageArea']<1200]
train = train[train['1stFlrSF']<2700]
train = train[train['2ndFlrSF']<1700]
train = train[train['TotalBsmtSF']<3200]




## 2-3. 상관관계 분석 (히트맵)

# 상관계수
numerical_columns = train.select_dtypes(include=['number'])  # 숫자 열만 선택
corr = numerical_columns.corr()
corr_columns = corr.index[abs(corr['SalePrice']) >= 0.4]  # 상관계수 0.4 이상만 포함
print(corr_columns)

# 히트맵
plt.figure(figsize=(13, 10))
heatmap = sns.heatmap(train[corr_columns].corr(), annot=True, cmap="coolwarm")
# plt.show()



## 2-3. 데이터 합치기 (concat)
df_train = train.drop(['SalePrice'], axis=1) #test데이터에 없는 예측해야될 값인 'SalePrice'컬럼을 train데이터에서 지워줌
df = pd.concat((df_train, test)) #데이터 합치기

## 2-4. 타겟변수 쏠림 현상 파악
sns.displot(train['SalePrice']) #그래프의 왼쪽으로의 쏠림현상 확인
# plt.show()

train['SalePrice'] = np.log1p(train["SalePrice"]) #로그 변환을 통해 정규성을 띄도록 바꿔줌
sns.displot(train['SalePrice']) #정규화 된 것을 확인
# plt.show()
price = train['SalePrice'] #변수에 할당



