# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (assuming you have a CSV file with sales data)
data = pd.read_csv("advertising.csv")

# Display the first few rows of the dataset
print(data.head())

# Assume 'TV', 'Radio', 'Newspaper' are the features and 'Sales' is the target variable
# Define features (X) and target variable (y)
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Now, you can use this trained model to predict sales for new data
# For example:
new_data = pd.DataFrame({'TV': [200], 'Radio': [80], 'Newspaper': [40]})
predicted_sales = model.predict(new_data)
print(f"Predicted Sales for new data: {predicted_sales[0]}")


from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Assume 'X' contains your original features and 'y' is the target variable

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Split the data into training and testing sets for polynomial features
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Now, you can proceed to model training using these polynomial features, for instance, with a linear regression model:

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the linear regression model
model_poly = LinearRegression()

# Train the model on the polynomial features
model_poly.fit(X_train_poly, y_train)

# Make predictions on the test set with polynomial features
predictions_poly = model_poly.predict(X_test_poly)

# Evaluate the model
mse_poly = mean_squared_error(y_test, predictions_poly)
r2_poly = r2_score(y_test, predictions_poly)

print(f"Mean Squared Error (Polynomial): {mse_poly}")
print(f"R-squared Score (Polynomial): {r2_poly}")

import matplotlib.pyplot as plt
import seaborn as sns  # Optional, for some visualizations

# Your code for data manipulation, model training, etc.

# Example scatter plot
x_values = [1, 2, 3, 4, 5]  # Replace with your x-axis data
y_values = [10, 15, 13, 18, 20]  # Replace with your y-axis data

plt.scatter(x_values, y_values)
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Title of the Plot')
plt.show()
