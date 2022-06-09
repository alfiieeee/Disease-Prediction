
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

st.write("""
# Weather Prediction in Seattle
This app predicts the **Weather** type in Seattle!
""")

image= Image.open("weather.jpg")
st.image(image, caption='© Google')

st.sidebar.header('Please select your parameter below!')

def user_input_features():
    precipitation = st.sidebar.slider('**Precipitation**', 0.00, 55.90, 27.95)
    temp_min = st.sidebar.slider('**Minimum Temperature**', -7.10, 18.30, 5.60)
    temp_max = st.sidebar.slider('**Maximum Temperature**', -1.60, 35.60, 17.00)
    wind = st.sidebar.slider('**Wind**', 0.40, 9.50, 4.95)
    data = {'Precipitation': precipitation,
            'Minimum Temperature': temp_min,
            'Maximum Temperature': temp_max,
            'Wind': wind}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('This is your selected parameter')
st.write(df)

weather = pd.read_csv("https://raw.githubusercontent.com/alfiieeee/Weather-Prediction/main/seattle-weather.csv")
X = weather.drop('weather', axis = 1)
Y = weather['weather']

classification = RandomForestClassifier()
classification.fit(X, Y)

prediction = classification.predict(df)
prediction_proba = classification.predict_proba(df)

st.subheader('Weather labels and their corresponding index number')
weather.target_names = ['drizzle','rain','sun', 'snow', 'fog']
st.table(weather.target_names)

st.subheader('The prediction of the weather')
#st.write(weather.target_names[prediction])
st.write(prediction)

st.subheader('Probability of weather prediction')
st.write(prediction_proba)

st.snow()
