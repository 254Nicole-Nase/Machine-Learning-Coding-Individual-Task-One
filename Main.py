import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = 'Nairobi Office Price Ex.csv'
data = pd.read_csv(file_path)

# Separate the feature (X) and target (y) variables
X = data['SIZE'].values
y = data['PRICE'].values

# Function to calculate Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient Descent function to update slope (m) and y-intercept (c)
def gradient_descent(X, y, m, c, learning_rate, epochs):
    N = len(y)  # Number of data points
    error_history = []  # Store error for each epoch

    for epoch in range(epochs):
        y_pred = m * X + c  # Predictions based on current slope and intercept
        error = y - y_pred  # Difference between actual and predicted values

        # Calculate current MSE and store it
        mse = mean_squared_error(y, y_pred)
        error_history.append(mse)

        # Calculate gradients
        dm = -(2 / N) * np.dot(X, error)
        dc = -(2 / N) * np.sum(error)

        # Update slope (m) and intercept (c)
        m = m - learning_rate * dm
        c = c - learning_rate * dc

        # Print the error at each epoch
        print(f"Epoch {epoch + 1}: MSE = {mse:.4f}")

    return m, c, error_history

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
plt.plot(X, m_final * X + c_final, color='red', label='Line of Best Fit')
plt.xlabel('Office Size (sq ft)')
plt.ylabel('Price (USD)')
plt.title('Office Size vs Price (Linear Regression)')
plt.legend()
plt.show()

# Predict the office price when the size is 100 sq. ft.
office_size = 100
predicted_price = m_final * office_size + c_final
print("predicted office price when the size is 100 sq. ft. =", predicted_price)
