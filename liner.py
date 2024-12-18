import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Title of the app
st.title('Linear Regression Model')

# File upload
st.sidebar.header('Upload your CSV file')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview", df.head())

    # Select X and Y features from the DataFrame
    st.sidebar.header('Select Features')
    x_features = st.sidebar.multiselect('Select independent features (X)', df.columns.tolist(), default=df.columns[0])
    y_feature = st.sidebar.selectbox('Select dependent feature (Y)', df.columns.tolist())

    if x_features and y_feature:
        # Prepare the data for training the model
        X = df[x_features]
        y = df[y_feature]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the results on the test set
        y_pred = model.predict(X_test)

        # Calculate performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display performance metrics
        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"R-squared: {r2:.4f}")

        # Display the coefficients
        st.write("Model Coefficients:")
        coefficients = pd.DataFrame(model.coef_, x_features, columns=['Coefficient'])
        st.write(coefficients)

        # Visualize the model's predictions
        st.subheader('Predictions vs Actual')
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], '--k')
        ax.set_xlabel('Actual values')
        ax.set_ylabel('Predicted values')
        ax.set_title('Linear Regression: Actual vs Predicted')
        st.pyplot(fig)
    else:
        st.write("Please select both X and Y features to train the model.")
