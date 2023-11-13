import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from prediction import get_prediction, ordinal_encoder

model = joblib.load(r'Model/deploy.joblib')

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="centered")


#creating option list for dropdown menu
#option_minute = ['0', '15', '30', '45']
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
option_junction = ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'Unknown', 'T Shape',
 'X Shape']
option_roadsurface = ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep'] 
option_light_condition = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
 'Darkness - lights unlit']



options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']

features = ['hour','minute','day_of_week','casualties','vehicles_involved','driver_age','driving_experience','junction_type','road_surface_conditions','light_condition']


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        
        hour = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
        minute = st.slider("Select Minute: ", 0, 60, value=0, format="%d")
        day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)
        casualties = st.slider("Casualities: ", 1, 8, value=0, format="%d")
        #accident_cause = st.selectbox("Select Accident Cause: ", options=options_cause)
        vehicles_involved = st.slider("Pickup Hour: ", 1, 7, value=0, format="%d")
        #vehicle_type = st.selectbox("Select Vehicle Type: ", options=options_vehicle_type)
        driver_age = st.selectbox("Select Driver Age: ", options=options_age)
        #accident_area = st.selectbox("Select Accident Area: ", options=options_acc_area)
        driving_experience = st.selectbox("Select Driving Experience: ", options=options_driver_exp)
        #lanes = st.selectbox("Select Lanes: ", options=options_lanes)
        junction_type = st.selectbox("Select junction Type: ", options=option_junction)
        road_surface_conditions = st.selectbox("Select Road Surface Condition: ", options=option_roadsurface)
        light_condition = st.selectbox("Select Light Condition: ", options=option_light_condition)

        
        submit = st.form_submit_button("Predict")


    if submit:
        #minute = ordinal_encoder(minute, option_minute )
        day_of_week = ordinal_encoder(day_of_week, options_day)        
        #accident_cause = ordinal_encoder(accident_cause, options_cause)
        #vehicle_type = ordinal_encoder(vehicle_type, options_vehicle_type)
        driver_age =  ordinal_encoder(driver_age, options_age)
        #accident_area =  ordinal_encoder(accident_area, options_acc_area)
        driving_experience = ordinal_encoder(driving_experience, options_driver_exp) 
        #lanes = ordinal_encoder(lanes, options_lanes)
        junction_type = ordinal_encoder(junction_type, option_junction)
        road_surface_conditions = ordinal_encoder(road_surface_conditions, option_roadsurface)
        light_condition = ordinal_encoder(light_condition, option_light_condition)


        data = np.array([hour,minute,day_of_week,casualties,vehicles_involved,
                         driver_age,driving_experience,junction_type,road_surface_conditions,light_condition]).reshape(1,-1)
     
        
        pred = get_prediction(data=data, model=model)
        severity = ['Slight Injury', 'Serious Injury', 'Fatal injury']
        

        st.write(f"The predicted severity is:  {severity[pred[0]]}")

if __name__ == '__main__':
    main()