import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

dataset = None
x = None
y = None

def get_data():
    global dataset, x, y
    dataset = pd.read_csv('Data.csv')
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

def clean_data():
    global dataset, x, y
    imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
    imputer = imputer.fit(x[:,1:3])
    x[:, 1:3] = imputer.transform(x[:, 1:3])

def main(): 
    global dataset, x, y

    get_data()
    clean_data()

    print(dataset)
    print(x)
    print(y)

if __name__ == '__main__':
    main()