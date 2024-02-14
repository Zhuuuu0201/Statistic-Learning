import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Users/guangjingzhu/Desktop/statistic learning/HW8/backorders.csv')

# (a) Drop the SKU column
data = data.drop('sku', axis=1)

# (b) Check and impute null values
null_columns = data.columns[data.isnull().any()]

for col in null_columns:
    mean_value = data[col].mean()
    data[col].fillna(mean_value, inplace=True)

# (c) Check and correct data types
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
# (d) Convert categorical columns to numerical values using One-Hot Encoding
# Identify categorical columns that need one-hot encoding

data[categorical_cols]=data[categorical_cols].replace({'Yes': 1, 'No': 0})

# (e) Standardize numerical data
# List of numerical columns to be standardized
# numeric_cols = data.select_dtypes(include=['int64','float64']).columns
# Initialize the StandardScaler
scaler = MinMaxScaler()

# Apply standardization to the numerical columns and replace the original values
data[numeric_cols] = scaler.fit_transform(data[numeric_cols]) 

# (f) Split the data into train and test sets (90:10)
# Assuming 'went_on_backorder' is the target variable
# data = data.astype(np.float64)

# X = data.drop('went_on_backorder_Yes', axis=1)
# y = data['went_on_backorder_Yes']
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
# Split the data into train and test sets (90:10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Build Model 1
model = Sequential()
model.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=SGD(), metrics=['accuracy'])

# Fit the model for 100 epochs
history = model.fit(X_train, y_train, epochs=100, validation_split=0.1)

# Plot train/development accuracy and loss against epochs
plt.plot(history.history['accuracy'], label='training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model 1 Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model 1 Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Print the model summary
model.summary()

# Report performance on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

""" Logistic regression and a single-layer neural network with a sigmoid activation function are closely related. 
The logistic regression model can be considered as a special case of a neural network where there is no hidden layer, 
and the output layer consists of a single neuron with a sigmoid activation function. """





# Model 2
model = Sequential()
model.add(Dense(15, input_dim=X_train.shape[1], activation='tanh'))
model.add(Dense(1, activation='sigmoid'))  # Second layer for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Fit the model for 100 epochs
history = model.fit(X_train, y_train, epochs=100, validation_split=0.1)

# Plot train/development accuracy and loss against epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model 2 Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model 2 Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Print the model summary
model.summary()

# Report performance on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}') """



""" """ Therefore, for Model 2, the second layer should have:
Number of nodes: 1 (for binary classification)
Activation function: Sigmoid """


 # Model 3
model = Sequential()
model.add(Dense(25, input_dim=X_train.shape[1], activation='tanh'))
model.add(Dense(15, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Fit the model for 100 epochs
history = model.fit(X_train, y_train, epochs=100, validation_split=0.1)

# Plot train/development accuracy and loss against epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model 3 Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model 3 Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Print the model summary
model.summary()

# Report performance on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
 """


""" from tensorflow.keras.regularizers import l2

# Model 4
model = Sequential()
model.add(Dense(25, input_dim=X_train.shape[1], activation='tanh', kernel_regularizer=l2(0.01)))
model.add(Dense(15, activation='tanh', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Fit the model for 100 epochs
history = model.fit(X_train, y_train, epochs=100, validation_split=0.1)

# Plot train/development accuracy and loss against epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model 4 Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model 4 Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Print the model summary
model.summary()

# Report performance on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
