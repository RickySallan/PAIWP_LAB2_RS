#LOAD THE DATA SET
import pandas as pd
# Assuming you've uploaded the CSV file to your GitHub repository
file_path = '/workspaces/PAIWP_LAB2_RS/Billionaires Statistics Dataset.csv'
df = pd.read_csv(file_path)
########################################################################################

#DATA PREPARATION

# Remove unnecessary columns and any rows with missing values
df_cleaned = df.drop(columns=['rank', 'personName', 'city', 'source', 'countryOfCitizenship', 'latitude_country', 'longitude_country', 'organization', 'selfMade', 'status', 'gender', 'birthDate', 'lastName', 'firstName', 'title', 'date', 'state', 'residenceStateRegion','gdp_country']).dropna()

# Convert categorical data to numerical data using one-hot encoding
df_encoded = pd.get_dummies(df_cleaned, columns=['category', 'country', 'industries'])

# Separate the features and the target variable
X = df_encoded.drop('finalWorth', axis=1)
y = df_encoded['finalWorth']
##################################################################################################

#MODEL TRAINING

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)
##################################################################################################

#HYPERPARAMETER TUNING

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

# Initialize GridSearchCV
ridge_model = Ridge()
grid_search = GridSearchCV(estimator=ridge_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)

# Perform hyperparameter tuning
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Use the best model
best_ridge_model = grid_search.best_estimator_

##################################################################################################

#EVALUATE MODEL

from sklearn.metrics import mean_squared_error

# Predict on the test set
y_pred = best_ridge_model.predict(X_test)

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
