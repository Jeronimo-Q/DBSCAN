import pandas as pd

Data = pd.read_csv("2023-01-avon-and-somerset-street.csv")
print("Primeros 5 Registros")
print(Data.head())

Data = Data.drop(
    columns=[
        'Context',
        'Last outcome category',
        'Reported by',
        'Falls within',
        'Location'
    ],
    errors='ignore'
)
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

print('----------------------------------------------------------------------------')
print("Resumen descriptivo de los datos")
print(Data.describe())

print('----------------------------------------------------------------------------')


print('----------------------------------------------------------------------------')
print("Eliminación de registros incompletos")
print('Dimensiones iniciales:', Data.shape)
Data = Data.dropna(subset=['Latitude', 'Longitude'])
Data = Data.dropna(subset=['Crime ID'])
print('Dimensiones finales:', Data.shape)
