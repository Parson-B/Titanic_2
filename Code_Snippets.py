

import pandas as pd
import numpy as np
#visualization packages
import matplotlib.pyplot as plt
import seaborn as sns



#import data
training = pd.read_csv('train.csv')
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