
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Weather Prediction in Seattle
This app predicts the **Weather** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Precipitation = st.sidebar.slider('Precipitation', 0, 55.9, 27.95)
    Tempt_min = st.sidebar.slider('Minimum Temperature', -7.1, 18.3, 5.6)
    Tempt_max = st.sidebar.slider('Maximum Temperature', -1.6, 35.6, 17)
    Wind = st.sidebar.slider('Wind', 0.4, 9.5, 4.95)
    data = {'Precipitation': Precipitation,
            'Minimum Temperature': Tempt_min,
            'Maximum Temperature': Tempt_max,
            'Wind': Wind}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

weather = pd.read_csv("https://raw.githubusercontent.com/alfiieeee/Weather-Prediction/main/seattle-weather.csv")
X = weather.drop('weather', axis = 1)
Y = weather['weather']

classification = RandomForestClassifier()
classification.fit(X, Y)

prediction = classification.predict(df)
prediction_proba = classification.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
weather.target_names = ['drizzle','rain','sun', 'snow', 'fog']
st.write(weather.target_names)

st.subheader('Prediction')
#st.write(weather.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
