import joblib
import pandas as pd
import numpy as np
print("HOUSE PRICE PREDICTION")
model=joblib.load("models/best_house_price_model.pkl")
scaler=joblib.load("models/scaler.pkl")
print("\nEnter House Details\n")
MedInc=float(input("Median Income:"))
HouseAge=float(input("House Age:"))
AveRooms=float(input("Average Rooms:"))
AveBedrms=float(input("Average Bedrooms:"))
Population=float(input("Population:"))
AveOccup=float(input("Average Occupancy:"))
Latitude=float(input("Latitude:"))
Longitude=float(input("Longitude:"))
input_data=pd.DataFrame({
"MedInc":[MedInc],
"HouseAge":[HouseAge],
"AveRooms":[AveRooms],
"AveBedrms":[AveBedrms],
"Population":[Population],
"AveOccup":[AveOccup],
"Latitude":[Latitude],
"Longitude":[Longitude]
})
scaled_data = scaler.transform(input_data)
prediction = model.predict(scaled_data)
print("Predicted House Price")
print(f"\nEstimated Price : ${prediction[0]*100000:.2f}")
model=joblib.load("models/house_price_model.pkl")
scaler=joblib.load("models/scaler.pkl")