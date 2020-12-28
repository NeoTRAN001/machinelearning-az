import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values # Get elements by ID
y = dataset.iloc[:, -1].values

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    remainder='passthrough'
)
x = np.array(ct.fit_transform(x), dtype=np.float)

print(x)