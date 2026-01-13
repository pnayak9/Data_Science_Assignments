#!/usr/bin/env python
# coding: utf-8

# ### 6. Deployment with Streamlit:

# In[7]:


#import streamlit ,pandas and pickle library
import streamlit as st
import pandas as pd
import pickle


# In[8]:


#load the model
model=pickle.load(open('log_reg.pkl','rb'))


# In[9]:


# Load and read the trained model
def user_input_parameters():
    Pclass= st.sidebar.selectbox('Select the Pclass',[1,2,3]) # Changed to 1,2,3 as per typical Pclass values
    Sex= st.sidebar.selectbox('Sex (Male: 0, Female: 1)',[0,1])# set Male as 1 and Female as 0
    Fare= st.sidebar.slider('Fare',0.0,500.0,30.0)#set slider range
    Age= st.sidebar.slider('Age',0,100,25)#set the age range
    # Modified: Use selectbox for Embarked and create one-hot encoded columns
    embarked_input = st.sidebar.selectbox('Embarked', ['S', 'C', 'Q'])
    embarked_Q_val = 1 if embarked_input == 'Q' else 0
    embarked_S_val = 1 if embarked_input == 'S' else 0
    #create dictionary
    data={
        'Pclass': Pclass,
        'Sex': Sex,
        'Age': Age,
        'SibSp': 0,
        'Parch': 0,
        'Fare': Fare,
        'Embarked_Q': embarked_Q_val,
        'Embarked_S': embarked_S_val
    }
    #create dataframe
    features= pd.DataFrame(data,index=[0])
    return features

st.title('Titanic Survival Prediction')
#call the function
df= user_input_parameters()
#find probability and predicted probability
pred_prob=model.predict_proba(df)
pred=model.predict(df)
#display predicted value when buuten pressed
button= st.button('Predict')
if button==True:
    st.subheader('Predicted')
    st.write('Eligible' if pred_prob[0][1]>=0.5 else 'Not Eligible')
    st.subheader('Pred_Probabilities')
    st.write(pred_prob)


# ### Deploy and Run
# - Download the **Model_Deployment** as executable script as **Model_Deployment.py** and place it in anaconda prompt location
# - Open the anaconda prompt and run **"streamlit run Dep_File.py"**
# - It will redirect to the **"Titanic Survival Prediction"** web page for predict the eligibility of the new entries

# In[ ]:




