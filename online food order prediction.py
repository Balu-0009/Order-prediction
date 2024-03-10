#!/usr/bin/env python
# coding: utf-8

# In[1]:


# libraries

import numpy as np
import warnings
import random
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier

# pipeline
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV


# In[4]:


data = pd.read_csv("onlinedeliverydata.csv")
data.head()


# In[5]:


data = data[["Age","Gender", "Marital Status","Occupation", "Monthly Income", "Educational Qualifications",
             "Family size","latitude", "longitude", "Pin code","Output"]].copy()
data.info()


# In[6]:


data.isnull().sum() # no empty values


# # Data  Analysis with Visualization

# In[7]:


plt.figure(figsize=(10,8))
plt.title("Online Food Order Decisions Based on the Age of the Customer")
sns.countplot(x='Age',data=data,hue='Output',palette="Set3");


# In[8]:


plt.figure(figsize=(10,8))
plt.title("Online Food Order Decisions Based on the Size of the Family")
sns.countplot(x='Family size',data=data,hue='Output',palette="Set3_r");


# In[9]:


plt.figure(figsize=(10,8))
plt.title("Online Food Order Decisions Based on the Educational Qualifications of the Customer")
sns.countplot(x='Educational Qualifications',data=data,hue='Output',palette="Set2");


# In[10]:


plt.figure(figsize=(10,8))
plt.title("Online Food Order Decisions Based on the Educational Qualifications by the Occupation of the Customer")
sns.countplot(x='Educational Qualifications',data=data,hue='Occupation',palette="Set2_r");


# In[11]:


female = len(data[data["Gender"] == 'Female'])
male = len(data[data["Gender"] == 'Male'])
data_gender = [female,male]
labels = ['Female', 'Male']

colors = sns.color_palette('pastel')[0:5] # seaborn color palette to use

plt.figure(figsize=(10,8))
plt.title("Distribution of Customer's Gender")
plt.pie(data_gender, labels = labels, colors = colors, autopct='%.0f%%')
plt.show()


# In[12]:


plt.figure(figsize=(10,8))
plt.title("Educational Qualifications Based on the Gender of the Customer")
sns.countplot(x='Occupation',data=data,hue='Gender',palette="Pastel2");


# In[13]:


reorder = data[data["Output"] == "Yes"]
gender_data = reorder['Gender'].value_counts()
label = gender_data.index
counts = gender_data.values

colors = sns.color_palette('pastel')[2:4]

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text = "Which Gender is More Likely to Order Online Again?")
fig.update_traces(hoverinfo="label+percent", textinfo="value",marker=dict(colors=colors))

fig.show()


# In[14]:


reorder = data[data["Output"] == "Yes"]
gender_data = reorder['Gender'].value_counts()
label = gender_data.index
counts = gender_data.values

colors = sns.color_palette('pastel')[2:4]

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text = "Which Gender is More Likely to Order Online Again?")
fig.update_traces(hoverinfo="label+percent", textinfo="value",marker=dict(colors=colors))

fig.show()


# In[15]:


reorder = data[data["Output"] == "Yes"]
status_data = reorder['Marital Status'].value_counts()
label = status_data.index
counts = status_data.values

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text = "What is the Marital Status of Customers?")
fig.update_traces(hoverinfo="label+percent", textinfo="value",marker=dict(colors=colors))

fig.show()


# # Preparing the Data

# In[16]:


data["Monthly Income"].unique()


# In[17]:


data["Gender"] = data["Gender"].map({"Male":0,"Female":1}) # male or female

data["Marital Status"] = data["Marital Status"].map({"Married":0,"Single":1,"Prefer not to say":2})


data["Occupation"] = data["Occupation"].replace(to_replace=["Employee","Self Employeed"], value=1) # employed
data["Occupation"] = data["Occupation"].replace(to_replace=["Student","House wife"], value=0) # unemployed


data["Educational Qualifications"] = data["Educational Qualifications"].map({"Graduate": 1, 
                                                                             "Post Graduate": 2, 
                                                                             "Ph.D": 3, "School": 4,
                                                                             "Uneducated": 5})


data["Monthly Income"] = data["Monthly Income"].replace(to_replace=["No Income"], value=0) # no income
data["Monthly Income"] = data["Monthly Income"].replace(to_replace=["Below Rs.10000",
                                                                   "More than 50000",
                                                                   "25001 to 50000",
                                                                   "10001 to 25000"], value=1)  # has an income
                                                                             
#data["Feedback"] = data["Feedback"].map({"Negative ":0,"Positive":1}) # negative or positive

data["Output"] = data["Output"].map({"No":0,"Yes":1}) # no or yes


# # Prediction Model

# In[18]:


X = data.drop('Output',axis=1)
y = data['Output']

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, 
                                                    random_state=101)

print("Shape of train dataset : ", X_train.shape)
print("Shape of test dataset : ", X_test.shape)


# In[19]:


rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

print(rfc.score(X_test, y_test))


# In[20]:


pred = rfc.predict(X_test)

print(confusion_matrix(pred,y_test))

plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(pred,y_test), annot=True);


# In[21]:


# Evaluating a score by cross-validation
# cv determines the cross-validation splitting strategy
scores = cross_val_score(rfc, X_train, y_train,cv=5)

# average score
print("Accuracy: ", scores.mean(), scores.std() * 2)


# # Parameter Search

# In[22]:


parameters = {
    'randomforestclassifier__n_estimators': (20, 50, 100)
}

pipeline = make_pipeline(RandomForestClassifier())

pipeline


# In[23]:


gridsearch = GridSearchCV(pipeline, parameters, verbose=1, n_jobs= -1)


# In[24]:


gridsearch.fit(X_train,y_train)


# In[25]:


print("Best score %0.3f" % gridsearch.best_score_)
print("Best parameters set: ")
best_parameters = gridsearch.best_estimator_.get_params()

for params in sorted(parameters.keys()):
    print("\t%s: %r" % (params, best_parameters[params]))


# # Given the following customer, would this person order?

# In[26]:


r = random.randint(0,len(data))
print(r)
new_customer = data.drop('Output', axis=1).iloc[r]
new_customer


# In[27]:


rfc.predict(new_customer.values.reshape(1,10)) # predicted output


# In[29]:


data.iloc[r]['Output'] # actual Output


# In[ ]:




