# Machine Learning Coding Individual Task One

## Category  
**Regression**

## Sub-Category  
**Linear Regression**

## ML Model  
**Predictive Models**

## Problem Definition  
This task involves building a linear regression model to predict office prices in Nairobi using a dataset with one feature, **Office Size (sq ft)**, and one target, **Price (USD)**. The objective is to:  
1. Write two Python functions:
   - **Mean Squared Error (MSE)**: Used to evaluate the performance of the model.
   - **Gradient Descent**: An algorithm to update the slope and intercept values iteratively.
2. Set **random initial values** for slope (m) and y-intercept (c).
3. Train the model using gradient descent for **10 epochs** and display the **error at each epoch**.
4. Plot the **line of best fit** after training.
5. Use the trained model to **predict the price of an office of 100 sq. ft.**.

## Dataset  
The dataset used is **Nairobi Office Price Ex.csv**, containing:
- **Feature (X)**: Office size in sq. ft.
- **Target (y)**: Office price in USD.

## Code Overview  
### Key Functions

#### Mean Squared Error (MSE)  
This function measures the difference between the actual and predicted values.
```python
def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse
