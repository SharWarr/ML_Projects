#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd


# In[48]:


df = pd.read_csv("Obesity Classification.csv")


# In[49]:


df.head()


# In[50]:


df.shape


# In[51]:


df.isna().sum()


# In[52]:


df.describe()


# In[53]:


import seaborn as sns


# In[54]:


sns.boxplot(x="Age",data=df)


# In[55]:


df["Age"].unique()


# In[56]:


sns.boxplot(x="Height",data=df)


# In[57]:


sns.boxplot(x="BMI",data=df)


# In[58]:


sns.barplot(x=df.Label.value_counts().index, y=df.Label.value_counts())


# In[59]:


df["Height"].dtype


# In[60]:


sns.displot(df["Height"])


# In[61]:


sns.displot(df["Weight"])


# In[62]:


sns.displot(data=df, x="BMI", hue="Gender")


# Males seem to have higher BMI

# We will create Logistic Regression model to predict if a person will be over weight or not

# In[63]:


df['Label'] = df['Label'].apply(lambda x: 1 if x=="Overweight" else 0)


# In[64]:


df.head()


# In[65]:


sns.barplot(x=df.Label.value_counts().index, y=df.Label.value_counts())


# In[147]:


Obese = df[df["Label"] == 1]
NotObese = df[df["Label"] == 0]


# In[149]:


Obese.shape


# In[150]:


NotObese.shape


# In[151]:


# upsample minority class
Obese_upsampled = resample(Obese,replace=True,    # sample with replacement
                                 n_samples= 88, # to match majority class
                                 random_state=42)  # reproducible results


# In[152]:


Obese_upsampled.shape


# Now data imbalance is handled, lets join them

# In[153]:


df_upsampled = pd.concat([Obese_upsampled,NotObese])


# There is data imbalance that needs to be handled

# In[154]:


df_upsampled


# In[155]:


df_upsampled.shape


# In[156]:


sns.barplot(x=df_upsampled.Label.value_counts().index, y=df_upsampled.Label.value_counts())


# In[158]:


X = df_upsampled[["ID","Age","Height","Weight","BMI","Female","Male"]]


# In[159]:


y = df_upsampled["Label"]


# In[160]:


X.head()


# In[161]:


y.head()


# In[162]:


from sklearn.model_selection import train_test_split


# In[188]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


# In[189]:


X_train


# In[190]:


X_test


# In[191]:


y_train


# In[192]:


y_test


# In[193]:


y_train


# 
# Now the imbalance is rectified. Now we can perform Logistic Regression.

# In[194]:


from sklearn.linear_model import LogisticRegression


# In[195]:


logreg = LogisticRegression(random_state=16,max_iter=1000)


# We have gender as a categorical value , hence we need to convert that to numerical. Since it has only few categories we can use 
# One Hot encoding

# In[196]:


X_train


# In[198]:


y_test.shape


# In[199]:


X_test.shape


# In[215]:


# fit the model with data
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)


# In[201]:


y_pred


# In[202]:


from sklearn.metrics import confusion_matrix


# In[204]:


confusion_matrix(y_test, y_pred)


# In[207]:


from sklearn import metrics
confusion = metrics.confusion_matrix(y_test, y_pred)
print(confusion)
# Let's check the overall accuracy.
print(metrics.accuracy_score(y_test, y_pred))


# In[211]:


X_test


# In[219]:


y_pred_1 = logreg.predict([[84,29,150,30,13.3,1,0]])


# In[223]:


y_pred_1


# In[238]:


from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

app = FastAPI()

class request_body(BaseModel):
    ID : int
    Age : int
    Height: int
    Weight: int
    BMI : int
    Female : int
    Male : int
        
@app.post('/predict')
def predict(data : request_body):
    test_data = [[
            data.ID, 
            data.Age, 
            data.Height, 
            data.Weight,
            data.BMI,
            data.Female,
            data.Male
    ]]
    class_idx = logreg.predict(test_data)[0]
    return { 'class' : class_idx}


# In[ ]:




