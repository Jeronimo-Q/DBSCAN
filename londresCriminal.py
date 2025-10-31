import pandas as pd

Data = pd.read_csv("london_crime_by_lsoa.csv")
print("Primeros 5 Registros")
print(Data.head())
print('----------------------------------------------------------------------------')
print("Número de filas, columnas y tipo de datos")
print(Data.info())

print('----------------------------------------------------------------------------')
print("Valores faltantes por columna:")
faltantes = Data.isnull().sum()
print(faltantes)

print('----------------------------------------------------------------------------')
print("Valores Duplicados")
duplicated = Data.duplicated().sum()
print(duplicated)
#No existen registros duplicados por lo que no se tienen que eliminar

print('----------------------------------------------------------------------------')
print("Resumen descriptivo de los datos")
print(Data.describe())

print('----------------------------------------------------------------------------')


print('----------------------------------------------------------------------------')
print("Eliminación de registros incompletos")
print('Dimensiones iniciales:', Data.shape)
Data = Data.dropna()
print('Dimensiones finales: ', Data.shape)