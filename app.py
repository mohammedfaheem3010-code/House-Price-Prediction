import streamlit as st
import pandas as pd
import joblib
import os
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)
st.title("🏠 House Price Prediction System")
st.write(
    """
Welcome to the House Price Prediction Project.
This application predicts California House Prices using Machine Learning.
"""
)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Prediction",
        "Dataset",
        "Graphs",
        "Model Performance",
        "About"
    ]
)
if page=="Home":
    st.header("🏠 California House Price Prediction")
    st.image(
        "graphs/08_map.png",
        use_container_width=True
    )
    st.write("""
This project predicts California House Prices.
Algorithms Used
• Linear Regression
• Decision Tree
• Random Forest
• Gradient Boosting
Best Model
Random Forest
""")

df = pd.read_csv("data/housing.csv")
if page=="Dataset":
    st.header("Dataset")
    st.dataframe(df.head(20))
    st.write("Rows :",df.shape[0])
    st.write("Columns :",df.shape[1])
    st.write(df.describe())
    
model=joblib.load("models/best_house_price_model.pkl")
scaler=joblib.load("models/scaler.pkl")
medinc=st.number_input("Median Income",0.0,20.0,5.0)
houseage=st.number_input("House Age",1,60,20)
rooms=st.number_input("Average Rooms",1.0,20.0,5.0)
bedrooms=st.number_input("Average Bedrooms",0.5,5.0,1.0)
population=st.number_input("Population",1,50000,1000)
occupancy=st.number_input("Average Occupancy",1.0,20.0,3.0)
latitude=st.number_input("Latitude",32.0,42.0,34.0)
longitude=st.number_input("Longitude",-125.0,-114.0,-118.0)
if st.button("Predict Price"):
    input_data = pd.DataFrame({
        "MedInc":[medinc],
        "HouseAge":[houseage],
        "AveRooms":[rooms],
        "AveBedrms":[bedrooms],
        "Population":[population],
        "AveOccup":[occupancy],
        "Latitude":[latitude],
        "Longitude":[longitude]
    })
    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)
    st.success(
        f"Estimated House Price : ${prediction[0]*100000:,.2f}"
    )

if page=="Graphs":
    st.header("EDA Graphs")
    files = sorted(os.listdir("graphs"))
    for graph in files:
        st.image(
            f"graphs/{graph}",
            caption=graph,
            use_container_width=True
        )
if page=="Model Performance":
    st.header("Model Comparison")
    result = pd.read_csv("models/model_comparison.csv")
    st.dataframe(result)
    st.bar_chart(
        result.set_index("Model")["R2 Score"]
    )
if page=="About":
    st.header("About Project")
    st.write("""
House Price Prediction
Machine Learning Internship Project
Dataset
California Housing Dataset
Algorithms
✔ Linear Regression
✔ Decision Tre
✔ Random Forest
✔ Gradient Boosting
Developer
Mohammed Faheem
""")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Dataset Rows", "20,640")
with col2:
    st.metric("Features", "8")
with col3:
    st.metric("Models", "4")