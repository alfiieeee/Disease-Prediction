
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
   
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)
        
st.write("""
# Weather Prediction in Seattle
This app predicts the **Weather** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    precipitation = st.sidebar.slider('Precipitation', 0.00, 55.90, 27.95)
    temp_min = st.sidebar.slider('Minimum Temperature', -7.10, 18.30, 5.60)
    temp_max = st.sidebar.slider('Maximum Temperature', -1.60, 35.60, 17.00)
    wind = st.sidebar.slider('Wind', 0.40, 9.50, 4.95)
    data = {'Precipitation': precipitation,
            'Minimum Temperature': temp_min,
            'Maximum Temperature': temp_max,
            'Wind': wind}
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
