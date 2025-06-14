import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

def generate_house_data(n_samples=100):
    np.random.seed(50)
    size = np.random.normal(1400, 300, n_samples)
    price = size * 200 + np.random.normal(0, 10000, n_samples)
    return pd.DataFrame({'Size': size, 'Price': price})

def train_model():
    df=generate_house_data(100)
    X= df[['Size']]
    Y = df['Price']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def main():
    st.title("House Price Prediction")
    st.write("This app predicts house prices based on size using a simple linear regression model.")

    model = train_model()
    size = st.number_input("Enter the size of the house (in square feet):", min_value=500, max_value=5000, value=1500)

    if st.button("Predict Price"):
        predicted_price = model.predict([[size]])
        st.success(f"The predicted price for a house of size {size} sq ft is ${predicted_price[0]:,.2f}")

    df = generate_house_data()
    fig = px.scatter(df, x='Size', y='Price', title='House Size vs Price')
    fig.add_scatter(
        x=[size], y=[predicted_price[0]], 
        mode='markers', 
        marker=dict(color='red', size=10), 
        name='Predicted Price'
    )
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()