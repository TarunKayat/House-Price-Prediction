import streamlit as st
import pandas as pd
import numpy as np
  
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
 

# Load the dataset
df = pd.read_csv('Bengaluru_House_Data.csv')

# Data cleaning
df = df.dropna()
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

def convert_sqft(x):
    try:
        return float(x)
    except:
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df = df.dropna(subset=['total_sqft'])

# Prepare model data
df_model = df[['total_sqft', 'bath', 'bhk', 'price']]
X = df_model.drop('price', axis=1)
y = df_model['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Bengaluru House Price Predictor")

st.write("Enter the details of the house to predict the price (in Lakhs):")

sqft = st.number_input("Total Square Feet", min_value=300.0, max_value=10000.0, value=1000.0)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, value=2)

if st.button("Predict Price"):
    input_data = np.array([[sqft, bath, bhk]])
    predicted_price = model.predict(input_data)[0]
    st.success(f"Predicted Price: ₹ {predicted_price:.2f} Lakhs")
