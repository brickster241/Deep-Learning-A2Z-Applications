# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split



# IMPORT DATASET
dataset = pd.read_csv("Data.csv")

# IMPORT DEPENDENT AND INDEPENDENT VARIABLES
# X - INDEPENDENT VARIABLES
# y - DEPENDENT VARIABLE
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print(X, y)

# TAKE CARE OF MISSING DATA - (REMOVE NAN)
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# print(X)

# ENCODING CATEGORICAL DATA - (ONE-HOT-ENCODING , LABEL-ENCODING)

# ENCODE THE INDEPENDENT VARIABLE - (France, Spain, Germany)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print(X)

# ENCODE THE DEPENDENT VARIABLE
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

# SPLITTING THE DATASET INTO TRAINING SET AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# print(X_train, X_test, y_train, y_test)


# FEATURE SCALING
# SHOULD WE APPLY BEFORE SPLITTING OR AFTER SPLITTING THE DATASET ? - AFTER


# STANDARDIZATION = (x - mean(x) / SD(x))   -> Convert values in range from -3 to 3
# NORMALIZATION = (x - min(x) / max(x) - min(x))     -> Convert values from 0 to 1
sc = StandardScaler()
# NO STANDARDIZATION WILL BE DONE ON DUMMY VARIABLES

X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

