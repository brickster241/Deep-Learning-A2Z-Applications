# Using Artificial Neural Networks to predict customer leaving expectancy

# IMPORT THE LIBRARIES
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# PART 1 - DATA PREPROCESSING

# IMPORT THE DATASET
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# ENCODING CATEGORICAL DATA

# Label Encoding the "Gender" Column
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
# print(X)

# One Hot Encoding the "Geography" Column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print(X)

# Splitting the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling - Always Apply in Deep Learning
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(X_train, X_test, y_train, y_test)


# PART 2 - BUILDING THE ANN
# Initialising the ANN
ANN = tf.keras.models.Sequential()

# Adding the input layer and first hidden layer
ANN.add(tf.keras.layers.Dense(units=6, activation="relu"))

# Adding the second hidden layer
ANN.add(tf.keras.layers.Dense(units=6, activation="relu"))

# Adding the output layer
ANN.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# PART 3 - TRAINING THE ANN
# Compile the ANN
ANN.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# Binary function - binary_crossentropy
# Non Binary - categorical_crossentropy, activation - "softmax"

# Training the ANN on the training set
ANN.fit(X_train, y_train, batch_size=32, epochs=100)

# PART 4 - MAKING PREDICTIONS AND EVALUATING THE MODEL
"""
Homework:
Use our ANN model to predict if the customer with the following information will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?
"""
# Use the Predict Method - Must always be in an double pair of square brackets, Apply feature scaling
# Only use transform method, use 0.5 as we used sigmoid activation function.
predicted_prob = ANN.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
print("\n\nThe Predicted Probability is : {}".format(*predicted_prob[0]))

# Predicting the Test Set Results
y_pred = ANN.predict(X_test)
y_pred = (y_pred > 0.5)

# To check values of y_pred and y_test side by side, we can use the concatenate function.
# np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)

print("\nConfusion Matrix : \n")
print(cm)
print("\n\nAccuracy Score : {} %".format(score * 100))
