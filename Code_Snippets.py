
#Set local working directory 
import os 
import getpass

#Set working directory
os.getcwd()
username = getpass.getuser()
os.chdir('C:/Users/'+username+'/Desktop/PortableGit/Titanic_2')

#####################################################################


import pandas as pd
import numpy as np
#visualization packages
import matplotlib.pyplot as plt
import seaborn as sns



#import data
training = pd.read_csv('Data/train.csv')
print(training.head())

#grouping data
surv_by_sex = training['Survived'].groupby([training['Sex'], training['Pclass']], axis = 0).mean()
print(surv_by_sex)

#pivoting data
piv = pd.pivot_table(training, index = 'Survived',  aggfunc=np.sum)
print(piv)


#Barplot for surv_by_sex
#First extract data from group by row-wise
tmp=[['Sex','female','male']]
for row in surv_by_sex.unstack().transpose().iterrows():
    index, data = row
    tmp.append([index]+data.tolist())
Res=np.array(tmp)

#Create sets for m/f for bars C as pass. class
f = Res[:,1][1:].astype(np.float)
m = Res[:,2][1:].astype(np.float)
C = Res[:,0][1:].astype(np.float)
C_lab = ['1st Class', '2nd Class', '3rd Class'] #C_lab = ['Class ' + str(int(l))  for l in C]

#Create plot
fig, axes = plt.subplots(nrows=1, ncols=1)
ax=axes
#ax0, ax1, ax2, ax3 = axes.flatten()
colors = ['red', 'blue']

scal = 0.2
f_bar = ax.bar(C-.5*scal, f, width=scal, color='r', align='center')
m_bar = ax.bar(C+.5*scal, m, width=scal, color='b', align='center')

plt.xticks(C, C_lab)
ax.set_ylabel('Survival Percentage')

ax.legend( (f_bar[0],m_bar[0]), ('female','male'))
fig.suptitle('Survival Rate by Sex',fontsize=20)

fig.savefig('Plots/BarPlot_surv_by_sex.png')


#Plot to show survival against Fare
surv_by_fare = training['Survived'].groupby([training['Fare']],axis=0).agg({'Mean' : 'mean', 'Count' : 'count'})
print(surv_by_fare)

tmp=[['Fare','Survival Pct','Count']]
#tmp=[]
for row in surv_by_fare.iterrows():
    index, data = row
    tmp.append([index] + data.tolist())
Res = pd.DataFrame(tmp[1:],columns=tmp[0])

sns.jointplot(x='Fare', y='Survival Pct', data=Res).savefig('Plots/JointPlot_surv_by_fare.png')
sns.jointplot(x='Fare', y='Survival Pct', data=Res, kind='hex',gridsize=10).savefig('Plots/JointPlot_surv_by_fare_hex.png')


#Check the sample size - possibly identify the need for over sampling
training.Survived.value_counts()

sns.countplot(x='Survived',data=training,palette='hls')
plt.title('Absolute Survivors and Deceased')
plt.ylabel('Count')
plt.savefig('Plots/Survived_counts.png')


# Categories ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'] 
# Examine 'Survived' and consider first ['Pclass', 'Sex', 'Age', 'Fare']
for cat in ['Pclass', 'Sex', 'Age', 'Fare']:
    print(training[cat].unique())

#Analyse absolut survival numbers by class
pd.crosstab(training.Pclass,training.Survived).plot(kind='bar')
plt.title('Survival Number by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Number of Survivors')
plt.savefig('Plots/BarPlot_surv_by_class_abs.png')

#Show stacked as percentage
table=pd.crosstab(training.Pclass,training.Survived)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart Survival by Class')
plt.xlabel('Passgenger Class')
plt.ylabel('Proportion')
plt.savefig('Plots/StackedBarPlot_surv_by_class_pct.png')

#Create Passenger Age Histogram
training.Age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('Plots/Histogram_Age.png')

#Analyse absolut survival numbers by Age
#First create age cohorts in intervalls of ten years
tmp_df=training[['Age','Survived']]
tmp_age_groups=pd.DataFrame([(tmp_df['Age'] // 10) * 10]).transpose()
tmp_age_groups.columns=['Age_Group']
tmp_df=pd.concat([tmp_df,tmp_age_groups],axis=1)
#Since NaNs are not shown replace them by -1
tmp_df['Age_Group'] = tmp_df['Age_Group'].fillna(-1)

pd.crosstab(tmp_df['Age_Group'],tmp_df['Survived']).plot(kind='bar')
plt.title('Survival Number by Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Counts')
plt.savefig('Plots/BarPlot_surv_by_Age_Groups_abs.png')


#Applying tutorial from https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8?gi=e0234e9e6c9e

#Perform a binary logistic regression to indentify the prediction quality of features
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Examine 'Survived' and consider first ['Pclass', 'Sex', 'Age', 'Fare']
mod_training = training[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].copy()
mod_training['Age'] = mod_training['Age'].fillna(-1)
mod_training['Sex'].replace({'female':0, 'male':1},inplace=True)

#If there were significantly fewer survivors than deceased we could over sample their number
X = mod_training.loc[:, mod_training.columns != 'Survived']
y = mod_training.loc[:, mod_training.columns == 'Survived']

#If imblearn is not installed do so using pip:
# import pip
# pip.main(['install','imbalanced-learn'])
from imblearn.over_sampling import SMOTE

OS = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns

OS_data_X,OS_data_y=OS.fit_sample(X_train, np.ravel(y_train))
OS_data_X = pd.DataFrame(data=OS_data_X,columns=columns )
OS_data_y= pd.DataFrame(data=OS_data_y,columns=['Survived'])

print("length of oversampled data is ",len(OS_data_X))
print("Number of deceased in oversampled data",len(OS_data_y[OS_data_y['Survived']==0]))
print("Number of survivors",len(OS_data_y[OS_data_y['Survived']==1]))
print("Proportion of survivors in oversampled data: ",len(OS_data_y[OS_data_y['Survived']==0])/len(OS_data_X))
print("Proportion of deceased in oversampled data : ",len(OS_data_y[OS_data_y['Survived']==1])/len(OS_data_X))

#Recursive Feature Elimination
data_vars=mod_training.columns.values.tolist()
y_cols=['Survived']
X_cols=[i for i in data_vars if i not in y_cols]

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

rfe = RFE(logreg, 20)
rfe = rfe.fit(OS_data_X, OS_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

#Choose the rfe supporting categories
#Adapt for only one Sex {0,1} possible no floats
X=OS_data_X[X_cols].round({'Sex': 0})
y=OS_data_y['Survived']

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

#Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

#Diagonal is correct the others off
from sklearn import metrics
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Survival Prediction')
plt.legend(loc="lower right")
plt.savefig('Plots/Log_ROC.png')
plt.show()


#Next step is to implement an ANN (Artificial Neural Network) most likely using Gradient Descent

#Applying tutorial from https://towardsdatascience.com/building-your-own-artificial-neural-network-from-scratch-on-churn-modeling-dataset-using-keras-in-690782f7d051

#Removing Passenger name (no relevance for survival assumed) and Cabin number (for now)
X = training.iloc[:, [x for x in range(2,12) if (x != 3 and x!=10)]].values
y = training.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Making gender a number
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#Making ticket a number
labelencoder_X_2 = LabelEncoder()
X[:, 5] = labelencoder_X_2.fit_transform(X[:, 5])

#Making embarking port a number
#Make nans 0 first
X[:, 7] = np.array(['0' if x is np.nan else x for x in X[:, 7]], dtype=object)
labelencoder_X_3 = LabelEncoder()
X[:, 7] = labelencoder_X_3.fit_transform(X[:, 7])

#Removing unknown age nans 
X[:, 2] = np.array([-1.0 if str(x) == 'nan' else x for x in X[:, 2]], dtype=object)


onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras

from keras.models import Sequential

from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu', input_dim = 8))

classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)



