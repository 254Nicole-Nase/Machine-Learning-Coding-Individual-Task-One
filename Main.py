import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the provided CSV file
file_path = 'Nairobi Office Price Ex.csv'
nairobi_data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
nairobi_data.head()


# Extract the relevant columns (SIZE and PRICE)
nairobi_data_filtered = nairobi_data[['SIZE', 'PRICE']]

# Convert data to numpy arrays for efficient computation
X = nairobi_data_filtered['SIZE'].values
y = nairobi_data_filtered['PRICE'].values


# Function to calculate Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Function to perform Gradient Descent for Linear Regression
def gradient_descent(X, y, m, c, learning_rate, epochs):
    n = len(X)
    errors = []

    for epoch in range(epochs):
        # Prediction of y based on current slope (m) and intercept (c)
        y_pred = m * X + c

        # Compute MSE for this epoch
        error = mean_squared_error(y, y_pred)
        errors.append(error)

        # Calculate gradients for m and c
        dm = -(2 / n) * np.sum(X * (y - y_pred))
        dc = -(2 / n) * np.sum(y - y_pred)

        # Update m and c
        m = m - learning_rate * dm
        c = c - learning_rate * dc

        # Print the error at each epoch
        print(f"Epoch {epoch + 1}: MSE = {error:.4f}")

    return m, c, errors


# Set random initial values for m (slope) and c (y-intercept)
np.random.seed(42)  # For reproducibility
m_init = np.random.randn()  # Random slope
c_init = np.random.randn()  # Random intercept

# Training parameters
learning_rate = 0.0001  # Learning rate
epochs = 10  # Number of epochs

# Train the model using Gradient Descent
m_final, c_final, errors = gradient_descent(X, y, m_init, c_init, learning_rate, epochs)

# Plot the line of best fit after the final epoch
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, m_final * X + c_final, color='red', label='Best fit line')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price (in USD)')
plt.title('Linear Regression: Office Size vs. Price')
plt.legend()
plt.show()

# Predict the office price when size is 100 sq. ft
predicted_price = m_final * 100 + c_final
print("predicted office price when the size is 100 sq. ft. =", predicted_price)
