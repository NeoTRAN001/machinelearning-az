import numpy # Menejo de arreglos de n dimesiones
import pandas # Lectura y escritura de archivos CSV
from sklearn.impute import SimpleImputer # Manejo de datos faltantes en arreglos

dataset = None
x = None
y = None

def get_data():
    global dataset, x, y
    dataset = pandas.read_csv('People.csv')
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:,  -1].values

def clean_data():
    global x
    imputer = SimpleImputer(missing_values = numpy.nan, strategy = 'mean')
    imputer = imputer.fit(x[:, 1:3])
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