
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


#Perform a binary logistic regression to indentify the prediction quality of features
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

#Check for possible adaption to increase sample sizes
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
tmp_age_groups.columns=['Age Group']
tmp_df=pd.concat([tmp_df,tmp_age_groups],axis=1)

pd.crosstab(tmp_df['Age Group'],tmp_df['Survived']).plot(kind='bar')
plt.title('Survival Number by Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Number of Survivors')
plt.savefig('Plots/BarPlot_surv_by_Age_Groups_abs.png')














