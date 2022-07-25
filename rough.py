import streamlit as st
import pickle
import numpy as np
import pandas as pd
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import LabelEncoder
pipe=pickle.load(open('pipe.pkl','rb'))
dataset=pickle.load(open('dataset.pkl','rb'))
st.title('car rides prices predictor')
distance=st.selectbox('distance',np.arange(0,100))
cab_type=st.selectbox('Cab Type',dataset['cab_type'].unique())
Destination = st.selectbox('Destination', dataset["destination"].unique())
source=st.selectbox('Starting Point',dataset['source'].unique())
surge_multiplier=st.selectbox('surge_multiplier',dataset['surge_multiplier'].unique())
name=st.selectbox('Car Name',dataset['name'].unique())
if st.button('Predict price'):
    query=np.array([distance, cab_type, Destination, source, surge_multiplier, name])
    query=query.reshape(1,6)
    st.title(pipe.predict(query))
