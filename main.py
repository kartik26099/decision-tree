from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# fetch dataset from UCI ML Repository
iris = fetch_ucirepo(id=53)

# Print dataset information
print("Dataset Information:")
print(iris.data)

# Extract features and targets (as pandas dataframes)
x = iris.data.features.values
y = iris.data.targets

# Apply OneHotEncoder to the target variable y using ColumnTransformer
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough")
y_encoded = np.array(ct.fit_transform(y))

# Print the encoded target variable
print("\nEncoded Target Variable (y):\n", y_encoded)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=1)

# Initialize and train the decision tree regressor
regressor = DecisionTreeRegressor(random_state=50)
regressor.fit(x_train, y_train)

# Predict the target values for the test set
y_predict = regressor.predict(x_test)

# Print the predicted values
print("\nPredicted values (y_predict):\n", y_predict)

# Print the true values from the test set
print("\nTrue values (y_test):\n", y_test)

# Reverse the encoding to get back the original categorical values for predictions
onehot_encoder = ct.named_transformers_["encoder"]
y_predict_reverse = onehot_encoder.inverse_transform(y_predict)

# Print the reversed predictions
print("\nReversed Predicted values (y_predict_reverse):\n", y_predict_reverse)

# Reverse the encoding to get back the original categorical values for true values from the test set
y_reverted = onehot_encoder.inverse_transform(y_test)

# Print the reversed true values
print("\nReversed True values (y_reverted):\n", y_reverted)
