# importing all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Models from Scikit-learn
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

datas = pd.read_csv('final_test.csv')
datas = datas[datas['age'] >= 8]
print(datas)
print('shape is ')
print(datas.shape[0])
print(' ')
print('basic properties of our dataset : ')
print(' ')
print(datas.describe())
print(' ')
print('null values :')
print(" ")
print(datas.isna().sum())
print(" ")

#***********____data preprocessing____************

print('')
print("basic count plot of diffrent categories ")
print(' ')
fig, axes = plt.subplots(figsize=(20,10))
sns.countplot(x = 'size', data = datas,order=datas['size'].value_counts().index)
plt.show()
print(' ')
print("basic count plot of diffrent categories ")
print(' ')
fig, axes = plt.subplots(figsize=(20,10))
sns.countplot(x = 'age', data = datas,order=datas['age'].value_counts().index)
plt.show()
print(' ')
print("basic count plot of diffrent categories ")
fig, axes = plt.subplots(figsize=(20,10))
print(' ')
sns.countplot(x = 'height', data = datas,order=datas['height'].value_counts().index)
plt.show()
print(' ')
print("The above plot shows that most common  sizes is medium and least common size is xxl")
print(' ')
#sns.pairplot(data=datas, hue='size',diag_kind="hist" ,height=6)
#plt.show()
print(' ')
print("The above plots to show dependency of each feature on our categories ")
print(' ')

# boxplot
fig, axes = plt.subplots(1,3,figsize=(20,5))
fig.suptitle('categorical boxplot')
sns.boxplot(x = 'size',y = 'weight', data = datas,ax = axes[0])
axes[0].set_title('weight')
sns.boxplot(x = 'size',y = 'age', data = datas, ax = axes[1])
axes[1].set_title('age')
sns.boxplot(x = 'size',y = 'height', data = datas, ax = axes[2])
axes[2].set_title('height')
plt.show()
print('we observe from boxplot and the previous scatter plots that there  are sevral outliers that can reduce the predictablity of our model ')

#1. adding bmi coloum in dataframe
print(datas)
print(" ")
datas["bmi"] = (datas["weight"])/((datas["height"]/100)**2)
print('added bmi column in dataframe')
print(" ")
print(datas)
print(" ")
mean_s = datas.mean()# use them to calculate z score
std_s = datas.std()# use them to calculate z score
print(' ')
print(' ')
#2. removing age  under 8
datas = datas[datas['age'] >= 8]
print("removing age  under 8")
print(" ")
#2. removing age  over 90
datas = datas[datas['age'] <= 90]
print("removing age  over 90")
print(" ")
print("length of dataset")
print(" ")
print(datas.shape[0])
print(" ")
#4> Removing Outliers
#this step is taken in sevral more steps :
# replacing tha NaN values with mean
print("checking null values")
print(" ")
print('null values')
print(" ")
print(datas.isna().sum())
print(" ")
print("replacing NaN values with means")
datas["age"] = datas["age"].fillna(datas['age'].mean())
datas["height"] = datas["height"].fillna(datas['height'].mean())
datas["weight"] = datas["weight"].fillna(datas['weight'].mean())
datas["bmi"] = datas["bmi"].fillna(datas['bmi'].mean())
print(datas.describe())
print(" ")
print(datas.isna().sum())
print(" ")
# a. calculating z-score
print('zscore - dataframe')
print(' ')
datas['age'] = (datas['age']-mean_s['age']) /std_s['age']
datas['weight'] = (datas['weight']-mean_s['weight'])/std_s['weight']
datas['height'] = (datas['height']-mean_s['height'])/std_s['height']
datas['bmi'] = (datas['bmi']-mean_s['bmi'])/std_s['bmi']
print(datas)

#b. removing the z-scores of features  that are not inbetween the range (-3,3) with null value
#removing outliers
print("removing the z-scores of features  that are not inbetween the range (-3,3) with null value")
print(' ')
datas = datas[(datas['weight'] >= -3)&(datas['weight'] <= 3)]
datas = datas[(datas['age'] >= -3)&(datas['age'] <= 3)]
datas = datas[(datas['height'] >= -3)&(datas['height'] <= 3)]
datas = datas[(datas['bmi'] >= -3)&(datas['bmi'] <= 3)]
print(datas)
print('info abot z-score dataframe')
print(" ")
print(datas.describe())
print(' ')
print('total no of NaN values')
print(" ")
print(datas.isna().sum())
print(" ")
#c converting back to raw data, but the dataframe contains NaN
datas['age'] = mean_s['age'] + datas['age']*std_s['age']
datas['age'] = datas['age'].apply(np.ceil)
datas['weight'] = mean_s['weight'] + datas['weight']*std_s['weight']
datas['weight'] = datas['weight'].apply(np.ceil)
datas['height'] = mean_s['height'] + std_s['height']*datas['height']
datas['height'] = datas['height'].apply(np.ceil)
datas['bmi'] = mean_s['bmi'] + std_s['bmi']*datas['bmi']
print('orignal data with no outliers but with NaNs')
print(' ')
print(datas)
print(' ')
#checking null values
print("checking null values")
print(" ")
print('null values')
print(" ")
print(datas.isna().sum())
print(" ")
#d replacing tha NaN values with mean
datas["age"] = datas["age"].fillna(datas['age'].mean())
datas["height"] = datas["height"].fillna(datas['height'].mean())
datas["weight"] = datas["weight"].fillna(datas['weight'].mean())
datas["bmi"] = datas["bmi"].fillna(datas['bmi'].mean())

#converted orignal data wtih null values replaced with median

print('this is converted orignal dataset with no outliers and no nans')
print(datas)
print(' ')
print('checking null values : ')
print(" ")
print(datas.isna().sum())
print(" ")
print('details of final dataset')
print(" ")
print("countss")
print(" ")
print(datas.shape[0])
print(" ")
print(datas.describe())
# boxplot
fig, axes = plt.subplots(1,4,figsize=(20,5))
fig.suptitle('categorical boxplot')
sns.boxplot(x = 'size',y = 'weight', data = datas,ax = axes[0])
axes[0].set_title('weight')
sns.boxplot(x = 'size',y = 'age', data = datas, ax = axes[1])
axes[1].set_title('age')
sns.boxplot(x = 'size',y = 'height', data = datas, ax = axes[2])
axes[2].set_title('height')
sns.boxplot(x = 'size',y = 'bmi', data = datas, ax = axes[3])
axes[3].set_title('bmi')
print('we observe from boxplot that most of the  outliers are removed ')
plt.show()
#processing input traing and test data
def bmi_c(weight,height):
  height_m = height/100
  return weight/(height**2)
X = datas.drop("size", axis=1)
# Target
y = datas["size"]
X_train, X_test, y_train, y_test, = train_test_split(X,y, test_size=0.10)
# random forests
model = RandomForestClassifier(criterion ='entropy',n_estimators=100,max_depth=10)
model.fit(X_train, y_train)
pickle.dump(model, open('model.pkl', "wb"))
y_pred = model.predict(X_test)
score_rf = accuracy_score(y_test, y_pred)
print('Accuracy Score of Random Forest Classifier :', score_rf)
print(score_rf)