#!/usr/bin/env python
# coding: utf-8

# In[79]:


# Importing Libraries
       
import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline


# In[80]:


#Loading Dataset
train = pd.read_csv("Telecom Churn.csv")
test = pd.read_csv("Telecom Churn.csv")


# In[81]:


train.head(5)


# In[82]:


train.info()


# In[83]:


#Droping phone.number because it won't have any effect
train = train.drop('phone number',axis=1)
test = test.drop('phone number',axis=1)


# In[84]:


#All continous variables in cname
cont_name = train.columns[(train.dtypes=="float64")|(train.dtypes=="int64")].tolist()
print(cont_name)


# In[85]:


#All categorical var in cat_names and removing target var
cat_names = train.select_dtypes(exclude=np.number).columns.tolist()
cat_names.remove('churn')
cat_names


# In[86]:


train['international plan'].value_counts()


# In[87]:


#Checking missing values in train dataset
print(train.isnull().sum())  

#no missing value present in the train data


# In[88]:


#Checking missing values in test data set
print(test.isnull().sum())  

#no missing value present in the test data


# In[89]:


#Target Variable data distribution
plt.figure(figsize=(6,6))
sns.countplot(x=train.churn,palette='deep')
plt.xlabel('Customer Churn',fontsize=12)
plt.ylabel('Count',fontsize=12)
plt.title("Distribution of Customer Churning ",fontsize= 20)
plt.show()

# It is clear that here is a class Imbalance problem
# In[90]:


#Relationational bar graph for checking data distribution with respect to target variable
def diff_bar(x,y):
    
    train.groupby([x,y]).size().unstack(level=-1).plot(kind='bar', figsize=(35,10))
    plt.xlabel(x,fontsize= 25)
    plt.ylabel('count',fontsize= 25)
    plt.legend(loc=0,fontsize= 25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("{X} Vs {Y}".format(X=x,Y=y),fontsize = 40)
    plt.show()


# In[91]:


#State Wise Churning of customer
diff_bar('state','churn')


# In[92]:


#area_code Wise Churning of customer
diff_bar('area code','churn')


# In[93]:


#International_Plan Wise Churning of customer
diff_bar('international plan','churn')


# In[94]:


#No. of Customer Churning and had a Voice mail plan
diff_bar('voice mail plan','churn')


# In[95]:


#Number of Customer_Service Call Wise Churning of customer
diff_bar('customer service calls','churn')


# In[96]:


#Scatter plot function
def diff_scattr(x,y):
    fig=plt.figure()
    fig=sns.lmplot(x,y,data=train,fit_reg=False)
    plt.xlabel(x,fontsize=14)
    plt.ylabel(y,fontsize=14)
    plt.title("{X} and {Y} Scatter Plot".format(X=x,Y=y),fontsize = 16)
    plt.show()


# In[97]:


#Total intl charge and Total intl Minute
diff_scattr('total intl charge','total intl minutes')


# In[98]:


# Total night charge and Total night Minute
diff_scattr('total night charge','total night minutes')


# In[99]:


#Total eve charge and Total eve Minute
diff_scattr('total eve charge','total eve minutes')


# In[100]:


#Total day charge and Total Day Minute
diff_scattr('total day charge','total day minutes')


# In[101]:


#function for converting categoric to num codes for plotting box plot
def cat_to_num(data):
    for i in range(0, data.shape[1]):
        #print(i)
        if(data.iloc[:,i].dtypes == 'object'):
            data.iloc[:,i] = pd.Categorical(data.iloc[:,i])
            data.iloc[:,i] = data.iloc[:,i].cat.codes
            data.iloc[:,i] = data.iloc[:,i].astype('object')
    return data


# In[102]:


train = cat_to_num(train)
test = cat_to_num(test)


# In[103]:


#Plotting Box Plot
for i in cont_name:
    plt.figure()
    plt.clf() #clearing the figure
    sns.boxplot(train[i],palette="deep")
    plt.title(i)
    plt.show()


# In[104]:


#Treating Out Liers and Converting them to nan
for i in cont_name:
    q75, q25 = np.percentile(train.loc[:,i], [75 ,25])
    iqr = q75 - q25
    minn = q25 - (iqr*1.5)
    maxx = q75 + (iqr*1.5)
    train.loc[train.loc[:,i] < minn,i] = np.nan
    train.loc[train.loc[:,i] > maxx,i] = np.nan
    print('{var} -----> {X}   Missing'.format(var = i, X = (train.loc[:,i].isnull().sum())))


# In[105]:


# Replacing all the nan values with the average of that column
for i in cont_name:
    train = train.apply(lambda x:x.fillna(train.loc[:,i].mean()))


# In[106]:


train.head(5)


# In[107]:


#Setting up the pane or matrix size
f, ax = plt.subplots(figsize=(18,12))  #Width,height

#Generating Corelation Matrix
corr = train[cont_name].corr()

sns.heatmap(corr,mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220,10, as_cmap=True),            square=True, ax=ax,annot=True,linewidths=1 , linecolor= 'black',vmin = -1, vmax = 1)

plt.show()


# In[108]:


#checking Relation b/w categorical variables with respect to target var
from scipy.stats import chi2_contingency
for i in cat_names:
    print(i)
    #As we know imput to chi square is always a contiguency table so we generating it using crostab function present in pd
    chi2, p, dof, ex =chi2_contingency(pd.crosstab(train['churn'],train[i]))
    #as above pd.crosstab(dependent variable , independent variable)
    print(p)
    
#chi2 Actual chi square test value
#p = pvalue
#dof = degree of freedom
#ex = excepted value
# If p value is less than 0.05 then we will reject null hypothesis
#Null = both the variables are independent
#Alternate = Both the variables are not independent


# In[109]:


#Removing correlated variable & the variable which doesn't contain any meaningfull info
remv_col = ['state','total day charge','total eve charge','total night charge','total intl charge']
train = train.drop(remv_col,axis=1)
test = test.drop(remv_col,axis=1)


# In[110]:


#Updating values _after removal of var
cont_name = ['account length', 'number vmail messages', 'total day minutes', 'total day calls', 'total eve minutes',
         'total eve calls', 'total night minutes', 'total night calls', 'total intl minutes', 'total intl calls',
         'customer service calls']
print(cont_name)


# In[111]:


#All categorical var and removing target var
cat_names = ['area code', 'international plan', 'voice mail plan']
print(cat_names)


# In[112]:


#Checking distribution of data via pandas visualization
train[cont_name].hist(figsize=(20,20),color='g',alpha = 0.7)
plt.show()


# In[113]:


#Histogram breaks down by target variable
def plot_hist_y(x,y):
    plt.hist(list(x[y == 1]),color='green',label='True',bins='auto')
    plt.hist(list(x[y == 0]),color='grey', alpha = 0.7, label='False',bins='auto')
    plt.title("Histogram of {var} breakdown by {Y}".format(var = x.name,Y=y.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")
    plt.show()


# In[114]:


for i in cont_name:
    plot_hist_y(train[i],train.churn)


# In[120]:


#Applying standarization as most of the variables are normalized distributed
def scale_standard(data):
    for i in cont_name:
        print(i)
        data[i] = (data[i] - data[i].mean())/data[i].std()
    return data 


# In[122]:


#Standardizing Scale
train = scale_standard(train)
test = scale_standard(test)


# In[123]:


train.head()


# In[126]:


from sklearn.model_selection import train_test_split,RandomizedSearchCV

X = train.iloc[:,:14]
y = train.iloc[:,14]
y=y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# In[127]:


(X_train.shape),(y_train.shape)


# In[128]:


from imblearn.over_sampling import SMOTE

Smo = SMOTE(random_state=101)
X_train_res, y_train_res = Smo.fit_sample(X_train,y_train)


# In[129]:


(X_train_res.shape,y_train_res.shape)


# In[130]:



def pred(model_object,predictors,compare):
    predicted = model_object.predict(predictors)
    cm = pd.crosstab(compare,predicted)
    TN = cm.iloc[0,0]
    FN = cm.iloc[1,0]
    TP = cm.iloc[1,1]
    FP = cm.iloc[0,1]
    print("CONFUSION MATRIX ------->> ")
    print(cm)
    print()
    
    ##check accuracy of model
    print('Classification paradox :------->>')
    print('Accuracy :- ', round(((TP+TN)*100)/(TP+TN+FP+FN),2))
    print()
    print('Specificity //  True Negative Rate :- ',round((TN*100)/(TN+FP),2))
    print()
    print('Sensivity // True Positive Rate // Recall :- ',round((TP*100)/(FN+TP),2))
    print()
    print('False Negative Rate :- ',round((FN*100)/(FN+TP),2))
    print()
    print('False Postive Rate :- ',round((FP*100)/(FP+TN),2))
    print()
    print(classification_report(compare,predicted))


# In[131]:


# Random Forest
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
#Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100,random_state=101).fit(X_train_res,y_train_res)

#Model Score on Valdation Data Set
pred(rf_model,X_test,y_test)


# In[132]:


from sklearn.linear_model import LogisticRegression
#logistic without binaries
logit_model = LogisticRegression(random_state=101).fit(X_train_res,y_train_res)

#Model Score on Valdation Data Set
pred(logit_model,X_test,y_test)


# In[133]:


from sklearn.neighbors import KNeighborsClassifier
#KNN Model Development
KNN_Model = KNeighborsClassifier(n_neighbors=5).fit(X_train_res,y_train_res)

#Model Score on Valdation Data Set
pred(KNN_Model,X_test,y_test)


# In[134]:


from sklearn.naive_bayes import GaussianNB
#Navie Model Development
Naive_model = GaussianNB().fit(X_train_res,y_train_res)

#Model Score on Valdation Data Set
pred(Naive_model,X_test,y_test)


# # Final Model :- Random Forest
# As above random forest fits best for out dataset out of our tested models

# In[135]:


# Training Final Model With Optimum Parameters
final_Model = RandomForestClassifier(random_state=101, n_estimators = 500,n_jobs=-1)
final_Model.fit(X_train_res,y_train_res)


# In[136]:


#Calculating feature importances
importances = final_Model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::1]

# Rearrange feature names so they match the sorted feature importances
names = [train.columns[i] for i in indices]

# Creating plot
fig = plt.figure(figsize=(10,10))
plt.title("Feature Importance")

# Add horizontal bars
plt.barh(range(X.shape[1]),importances[indices],align = 'center')
plt.yticks(range(X.shape[1]), names)
plt.show()


# In[137]:


from sklearn.metrics import roc_curve,auc,roc_auc_score
# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(y_test, final_Model.predict_proba(X_test)[:,1])
# Calculate the AUC
roc_auc = auc(fpr, tpr)
print ('ROC AUC: %0.2f' % roc_auc)

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# # Final Test Data Predictions¶

# In[138]:


#Test Data Spliting parts target and Predictors
XX = test.iloc[:,:14].values  #predictors
yy = test.iloc[:,14].values   #target
yy=yy.astype('int')


# In[139]:


#Predicting test data 
#pred(model_object=final_Model,predictors=XX,compare=yy)

Churn_Pred = final_Model.predict(XX)
cm = pd.crosstab(yy,Churn_Pred)
TN = cm.iloc[0,0]
FN = cm.iloc[1,0]
TP = cm.iloc[1,1]
FP = cm.iloc[0,1]
print("CONFUSION MATRIX ----->> ")
print(cm)
print()
##check accuracy of model
print('Accuracy :- ', round(((TP+TN)*100)/(TP+TN+FP+FN),2))
print('False Negative Rate :- ',round((FN*100)/(FN+TP),2))
print('False Postive Rate :- ',round((FP*100)/(FP+TN),2))


# In[140]:


print(classification_report(yy,Churn_Pred))


# # AUC & ROC over Test Data

# In[141]:


from sklearn.metrics import roc_curve,auc,roc_auc_score
# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(yy, final_Model.predict_proba(XX)[:,1])
# Calculate the AUC
roc_auc = auc(fpr, tpr)
print ('ROC AUC: %0.2f' % roc_auc)

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




