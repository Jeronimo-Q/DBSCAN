import pandas as pd


def cargar_datos(ruta):

    data = pd.read_csv(ruta)
    print("Primeros 5 Registros")
    print(data.head())
    return data


def eliminar_columnas(data):
    columnas_a_eliminar = [
        'Context',
        'Last outcome category',
        'Reported by',
        'Falls within',
        'Location'
    ]
    data = data.drop(columns=columnas_a_eliminar, errors='ignore')
    return data


def explorar_datos(data):

    print("----------------------------------------------------------------------------")
    print("Número de filas, columnas y tipo de datos")
    print(data.info())
    print("----------------------------------------------------------------------------")
    print("Valores faltantes por columna:")
    print(data.isnull().sum())
    print("----------------------------------------------------------------------------")
    print("Valores Duplicados")
    print(data.duplicated().sum())
    print("----------------------------------------------------------------------------")
    print("Resumen descriptivo de los datos")
    print(data.describe())
    print("----------------------------------------------------------------------------")


def outliers(data):

    data = data.drop_duplicates()

    print("Eliminación de registros incompletos")
    print('Dimensiones iniciales:', data.shape)
    data = data.dropna(subset=['Latitude', 'Longitude', 'Crime ID'])

    Q1_lat = data['Latitude'].quantile(0.25)
    Q3_lat = data['Latitude'].quantile(0.75)
    IQR_lat = Q3_lat - Q1_lat

    Q1_lon = data['Longitude'].quantile(0.25)
    Q3_lon = data['Longitude'].quantile(0.75)
    IQR_lon = Q3_lon - Q1_lon

    outliers_iqr = data[
        ((data['Latitude'] < (Q1_lat - 1.5 * IQR_lat)) | (data['Latitude'] > (Q3_lat + 1.5 * IQR_lat))) |
        ((data['Longitude'] < (Q1_lon - 1.5 * IQR_lon)) | (data['Longitude'] > (Q3_lon + 1.5 * IQR_lon)))
    ]

    print("Outliers detectados por IQR:", len(outliers_iqr))
    data = data.drop(outliers_iqr.index)

    print('Dimensiones finales:', data.shape)
    return data


def normalizar_texto(data):
    """Limpia y estandariza los textos en columnas clave."""
    data['Crime type'] = data['Crime type'].str.strip().str.lower()
    data['LSOA name'] = data['LSOA name'].str.strip().str.lower()
    return data


def cleanData():
    ruta = "2023-01-avon-and-somerset-street.csv"

    Data = cargar_datos(ruta)

    Data = eliminar_columnas(Data)
    explorar_datos(Data)
    Data = outliers(Data)
    Data = normalizar_texto(Data)
    print("Primeros registros del dataset limpio:")
    print(Data.head())
    return Data
    #Data.to_csv("Crimes_UK_Clean.csv", index=False)
    #print("Dataset limpio guardado como 'Crimes_UK_Clean.csv'")



