import pandas as pd

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Explore the dataset
print(data.head())
print(data.info())
print(data['Class'].value_counts())  # Check class distribution


from sklearn.preprocessing import StandardScaler

# Separate features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from imblearn.over_sampling import SMOTE

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)


from sklearn.metrics import classification_report

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
print(classification_report(y_test, y_pred))

# Check for missing values
print(data.isnull().sum())

# Handle missing values (if any)
# For example, drop columns with too many missing values:
data_cleaned = data.dropna(axis=1)

# Feature engineering example: Creating a new feature 'hour' from the 'Time' column
data['hour'] = data['Time'] // 3600  # Convert seconds to hours




# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Load your sales dataset (replace 'sales_data.csv' with your file)
data = pd.read_csv('sales_data.csv')

# Display first few rows of the dataset and check its structure
print(data.head())
print(data.info())

# Assuming multiple features in the dataset
selected_features = ['Advertising_Expenditure', 'Target_Audience', 'Platform_Selection']
X = data[selected_features]  # Features
y = data['Sales']  # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize/Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the linear regression model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression Metrics:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Hyperparameter tuning for RandomForestRegressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3)
grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_

# Initialize and train the RandomForestRegressor with best parameters
best_rf_model = RandomForestRegressor(**best_params, random_state=42)
best_rf_model.fit(X_train_scaled, y_train)

# Make predictions on the test set using RandomForestRegressor
y_pred_rf = best_rf_model.predict(X_test_scaled)

# Evaluate the RandomForestRegressor model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Regression Metrics:")
print(f"Mean Squared Error: {mse_rf}")
print(f"R-squared: {r2_rf}")

# Predict future sales with new data using the RandomForestRegressor model
new_data = pd.DataFrame({
    'Advertising_Expenditure': [1000, 1500, 2000],
    'Target_Audience': [0.8, 0.6, 0.7],
    'Platform_Selection': [1, 2, 3]
})  # Example new data

new_data_scaled = scaler.transform(new_data)
predicted_sales_rf = best_rf_model.predict(new_data_scaled)
print("Predicted sales for new data using RandomForestRegressor:")
print(predicted_sales_rf)



