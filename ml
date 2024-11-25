import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('train.csv')

# Display first few rows of the data
print(df.head())

# Data Preprocessing

# Handle missing values
df.fillna(df.mean(), inplace=True)  # Fill numeric columns with mean
df.fillna(df.mode().iloc[0], inplace=True)  # Fill categorical columns with mode

# Convert categorical variables to numeric using Label Encoding
categorical_columns = df.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Feature Selection
features = df.drop(columns=['SalePrice'])
target = df['SalePrice']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the results
plt.figure(figsize=(10,6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'s':10}, line_kws={'color':'red'})
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Sale Prices')
plt.show()
