#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv(r"Train.csv")
df


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df['Discount_offered_%']=100*df['Discount_offered']/df['Cost_of_the_Product']


# In[7]:


df.head()


# In[8]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[9]:


comm= df.copy()
for col in comm.select_dtypes(include='O').columns:
    comm[col]=le.fit_transform(comm[col])


# In[10]:


comm.head()


# In[11]:


def summary(df,column):
    print("Distinct : ",len(pd.unique(df[column])))
    print("Missing : ",df[column].isnull().sum())
    print("Sum : ",df[column].sum())
    print("Mode : ",st.mode(df[column]))
    print("stddev : ",np.std(df[column]))
    print("CV :",100*(np.std(df[column]))/(np.mean(df[column]))) #coefficient of variation
    print("Min : ",df[column].min())
    print("Max : ",df[column].max())
    print("Mean : ",df[column].mean())
    print("Q1 : ",np.quantile(df[column],0.25))
    print("Q1 : ",np.quantile(df[column],0.5))
    print("Q1 : ",np.quantile(df[column],0.75))
    
    
def values(df,column):
    for i in df[column].unique():
        print(i)


def values_df(df):
    for i in df.columns:
        print(i)
        for j in df[i].unique():
            print(j)
        print("-"*20)
        
def proportion(df,column):
    for i in df[column].unique():
        counts = (sum(df[column]==i)/df[column].count()) *100
        print (i,' dengan proporsi {}%'.format(counts))


# In[12]:


df.describe()


# In[13]:


comm.describe()


# In[14]:


df.info()


# In[15]:


comm.info()


# In[16]:


proportion(comm,'Reached.on.Time_Y.N')


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[18]:


plt.figure(figsize=(16,9))
x = comm.drop(['Warehouse_block','Mode_of_Shipment','Product_importance','Gender'],axis = 1)
ax = sns.heatmap(comm.corr(),annot = True,cmap = 'viridis')
plt.show()


# In[19]:


plt.figure(figsize=(16,9))
x = comm
ax = sns.heatmap(comm.corr(),annot = True,cmap = 'viridis')
plt.show()


# In[20]:


df.info()


# In[21]:


comm.info()


# In[22]:


sns.set_theme(style='whitegrid')
sns.set(rc = {'figure.figsize':(13,8)})
df['Warehouse_block'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


# In[23]:


sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='Warehouse_block',hue='Reached.on.Time_Y.N',data=df,kind="count",height=8, aspect=1.5)


# In[24]:


sns.set_theme(style='whitegrid')
sns.set(rc = {'figure.figsize':(13,8)})
df['Mode_of_Shipment'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


# In[25]:


sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='Mode_of_Shipment',hue='Reached.on.Time_Y.N',data=df,kind="count",height=8, aspect=1.5)


# In[26]:


sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='Customer_care_calls',hue='Reached.on.Time_Y.N',data=df,kind="count",height=8, aspect=1.5)


# In[27]:


sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='Customer_rating',hue='Reached.on.Time_Y.N',data=df,kind="count",height=8, aspect=1.5)


# In[28]:


sns.histplot(data=df, x="Cost_of_the_Product",hue="Reached.on.Time_Y.N",element="step")


# In[29]:


sns.set()

#create boxplot in each subplot
sns.stripplot(data=df, x='Reached.on.Time_Y.N', y='Cost_of_the_Product')


# In[30]:


sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='Prior_purchases',hue='Reached.on.Time_Y.N',data=df,kind="count",height=8, aspect=1.5)


# In[31]:


sns.set_theme(style='whitegrid')
sns.set(rc = {'figure.figsize':(13,8)})
df['Product_importance'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


# In[32]:


sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='Product_importance',hue='Reached.on.Time_Y.N',data=df,kind="count",height=8, aspect=1.5)


# In[33]:


sns.set_theme(style='whitegrid')
sns.set(rc = {'figure.figsize':(13,8)})
df['Gender'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


# In[34]:


sns.set_theme(style="whitegrid")
plt.figure(figsize = (20,8))
sns.catplot(x='Gender',hue='Reached.on.Time_Y.N',data=df,kind="count",height=8, aspect=1.5)


# In[35]:


sns.set_theme(style='whitegrid')
sns.set(rc = {'figure.figsize':(13,8)})
df['Reached.on.Time_Y.N'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()


# In[36]:


sns.set()

#define plotting region (2 rows, 2 columns)

fig, axes = plt.subplots(2, 2)

#create boxplot in each subplot
sns.boxplot(data=df, x='Reached.on.Time_Y.N', y='Customer_care_calls', ax=axes[0,0])
sns.boxplot(data=df, x='Reached.on.Time_Y.N', y='Customer_rating', ax=axes[0,1])
sns.boxplot(data=df, x='Reached.on.Time_Y.N', y='Cost_of_the_Product', ax=axes[1,0])
sns.boxplot(data=df, x='Reached.on.Time_Y.N', y='Discount_offered_%', ax=axes[1,1])


# EDA

# In[37]:


comm = comm.drop(["ID"],axis=1)


# In[38]:


comm.head()


# In[39]:


comm.describe()


# In[40]:


Q12 = comm['Weight_in_gms'].quantile(0.25)
Q32 = comm['Weight_in_gms'].quantile(0.75)
IQR = Q32-Q12


# In[41]:


comm[comm['Weight_in_gms']>(Q32+(1.5*IQR))]


# Encodeing

# In[42]:


"""def one_hot_encoder(data,feature,keep_first=True):

    one_hot_cols = pd.get_dummies(data[feature])
    
    for col in one_hot_cols.columns:
        one_hot_cols.rename({col:f'{feature}_'+col},axis=1,inplace=True)
    
    new_data = pd.concat([data,one_hot_cols],axis=1)
    new_data.drop(feature,axis=1,inplace=True)
    
    if keep_first == False:
        new_data=new_data.iloc[:,1:]
    
    return new_data"""


# In[43]:


"""df_onehot=df.copy()
for col in df_onehot.select_dtypes(include='O').columns:
    df_onehot=one_hot_encoder(df_onehot,col)
    
df_onehot.head()"""


# feature scaling

# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
#from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,train_test_split
#from sklearn.metrics import accuracy_score, plot_confusion_matrix,classification_report, confusion_matrix,f1_score,roc_auc_score


# In[45]:


features = comm.drop('Reached.on.Time_Y.N',axis = 1)
target = comm['Reached.on.Time_Y.N']


X_train, X_test, y_train, y_test = train_test_split(features,target,test_size = 0.2,stratify=target)


# In[46]:


scaler = StandardScaler()
scaler.fit(X_train,y_train)


# In[47]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# k Near Neighnour

# In[48]:


knn_model = KNeighborsClassifier()
knn_model.fit(X_train,y_train)


# In[49]:


ypred_knn = knn_model.predict(X_test)
ypred_knn


# In[50]:


from sklearn.metrics import classification_report
print(classification_report(y_test,ypred_knn))


# Decision Tree

# In[51]:


dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)


# In[52]:


ypred_dt = dt_model.predict(X_test)
ypred_dt


# In[53]:


print(classification_report(y_test,ypred_dt))


# Logistic Regression

# In[54]:


model_logreg = LogisticRegression()
model_logreg.fit(X_train,y_train)


# In[55]:


ypred_logreg = model_logreg.predict(X_test)
ypred_logreg


# In[56]:


print(classification_report(y_test,ypred_logreg))


# Random forest 

# In[57]:


model_forest = RandomForestClassifier()
model_forest.fit(X_train,y_train)


# In[58]:


ypred_forest = model_forest.predict(X_test)
ypred_forest


# In[59]:


print(classification_report(y_test,ypred_forest))


# In[60]:


pip install xgboost


# Extreme Gradient Boosting 

# In[61]:


#import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)


# In[62]:


ypred_xgb = model_xgb.predict(X_test)
ypred_xgb


# In[63]:


print(classification_report(y_test,ypred_xgb))


# In[64]:


#pip install imblearn


# In[65]:


import imblearn
from imblearn.over_sampling import RandomOverSampler


# In[66]:


from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


# In[75]:


rus=RandomUnderSampler(random_state=1,replacement=True)
x_rus,y_rus=rus.fit_resample(features,target)


# In[76]:


features=x_rus
target=y_rus


# In[77]:


print("Original Dataset",Counter(target))
print("Resampled Dataset",Counter(y_rus))


# In[80]:


ros=RandomOverSampler(random_state=1)
x_ros,y_ros=ros.fit_resample(features,target)


# In[81]:


features=x_rus
target=y_rus


# In[83]:


print("Original Dataset",Counter(target))
print("Resampled Dataset",Counter(y_rus))


# In[86]:


from imblearn.over_sampling import SMOTE
xtrain,xtest,ytrain,ytest=train_test_split(features,target,test_size=0.30,random_state=1)


# In[87]:


lr=LogisticRegression()
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)
print(classification_report(ytest,ypred))


# In[89]:


counter=Counter(ytrain)
print("Before",counter)

sm=SMOTE()

xtrain,ytrain=sm.fit_resample(xtrain,ytrain)
xtrain1,ytrain1=sm.fit_resample(xtrain,ytrain)

counter=Counter(ytrain1)
print("after",counter)


# In[90]:


lr=LogisticRegression()
lr.fit(xtrain1,ytrain1)
ypred=lr.predict(xtest)
print(classification_report(ytest,ypred))


# In[91]:


knn_model = KNeighborsClassifier()
knn_model.fit(X_train,y_train)


# In[92]:


ypred_knn = knn_model.predict(X_test)
ypred_knn


# In[93]:


from sklearn.metrics import classification_report
print(classification_report(y_test,ypred_knn))


# In[94]:


dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)


# In[95]:


ypred_dt = dt_model.predict(X_test)
ypred_dt


# In[96]:


print(classification_report(y_test,ypred_dt))


# In[97]:


model_forest = RandomForestClassifier()
model_forest.fit(X_train,y_train)


# In[98]:


ypred_forest = model_forest.predict(X_test)
ypred_forest


# In[99]:


print(classification_report(y_test,ypred_forest))


# In[ ]:




