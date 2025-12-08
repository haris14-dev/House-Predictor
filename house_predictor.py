import os
os.system("pip install scikit-learn streamlit pandas numpy matplotlib")

import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import numpy as np

@st.cache_data
def train_model():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["output"] = data.target

    x = df.drop("output", axis=1)
    y = df["output"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, random_state=42
    )

    scale = StandardScaler()
    x_train_scaled = scale.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_train_scaled, y_train)

    return model, scale, data.feature_names

model, scale, feature_names = train_model()

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 California House Price Predictor")
st.write("Enter details below to predict the **median house value** for a California district.")

# Sidebar instructions
st.sidebar.header("ℹ️ About")
st.sidebar.write("""
This model uses the **California Housing Dataset** from scikit-learn  
and predicts the **median house value (in $100,000 units)**  
based on demographic and geographical data.
""")


st.subheader("Enter District Information:")

MedInc = st.number_input("Median Income (1 – 15)", min_value=0.5, max_value=15.0, value=5.0, step=0.1)
HouseAge = st.number_input("Average House Age (1 – 50)", min_value=1.0, max_value=50.0, value=20.0, step=1.0)
AveRooms = st.number_input("Average Rooms per Household (1 – 10)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
AveBedrms = st.number_input("Average Bedrooms per Household (0.5 – 5)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
Population = st.number_input("Population (200 – 10000)", min_value=200, max_value=10000, value=1500, step=100)
AveOccup = st.number_input("Average Occupancy (1 – 6)", min_value=1.0, max_value=6.0, value=3.0, step=0.1)
Latitude = st.number_input("Latitude (32 – 42)", min_value=32.0, max_value=42.0, value=34.0, step=0.1)
Longitude = st.number_input("Longitude (-125 – -114)", min_value=-125.0, max_value=-114.0, value=-118.0, step=0.1)

# Combine into dataframe
input_data = pd.DataFrame([[
    MedInc, HouseAge, AveRooms, AveBedrms,
    Population, AveOccup, Latitude, Longitude
]], columns=feature_names)


if st.button("🔮 Predict Price"):
    # Scale input data
    input_scaled = scale.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.success(f"🏡 **Predicted Median House Value:** ${prediction * 100000:,.0f}")
    st.caption(f"({prediction:.2f} × 100,000 USD)")

    # Optional display of user inputs
    st.write("### Your Inputs")
    st.dataframe(input_data)






