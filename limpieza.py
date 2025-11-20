import pandas as pd
import matplotlib.pyplot as plt

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

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

def grafico_dispersion_ultra(data_antes, data_despues):

    plt.figure(figsize=(9,8))

    # Estilo base elegante
    plt.style.use('seaborn-v0_8-darkgrid')

    # Colores pastel profesionales
    azul_pastel = "#5DADE2"
    rojo_pastel = "#EC7063"

    # --- CAPA SUAVE DE FONDO (GLOW) ---
    plt.scatter(
        data_antes['Longitude'],
        data_antes['Latitude'],
        s=25,
        c=azul_pastel,
        alpha=0.07,
        linewidths=0,
    )
    plt.scatter(
        data_despues['Longitude'],
        data_despues['Latitude'],
        s=25,
        c=rojo_pastel,
        alpha=0.07,
        linewidths=0,
    )

    # --- PUNTOS PRINCIPALES CON ESTÉTICA PREMIUM ---
    plt.scatter(
        data_antes['Longitude'],
        data_antes['Latitude'],
        s=10,
        c=azul_pastel,
        alpha=0.55,
        edgecolor='none',
        label='Antes de la limpieza'
    )
    plt.scatter(
        data_despues['Longitude'],
        data_despues['Latitude'],
        s=10,
        c=rojo_pastel,
        alpha=0.55,
        edgecolor='none',
        label='Después de la limpieza'
    )

    # Título estilizado
    plt.title(
        "Comparación Visual del Dataset\nAntes y Después del Proceso de Limpieza",
        fontsize=18,
        fontweight='bold',
        pad=15
    )

    # Etiquetas elegantes
    plt.xlabel("Longitude", fontsize=14)
    plt.ylabel("Latitude", fontsize=14)

    # Cuadrícula suave
    plt.grid(True, linestyle='--', alpha=0.3)

    # Leyenda estilo tarjeta
    legend = plt.legend(
        frameon=True,
        fontsize=12,
        fancybox=True,
        shadow=True,
        borderpad=1
    )
    legend.get_frame().set_facecolor('#F8F9F9')
    legend.get_frame().set_edgecolor('#D5D8DC')

    plt.tight_layout()

    plt.savefig("results/dispersion.jpg", dpi=350, bbox_inches='tight')
    plt.show()




def cleanData():
    ruta = "2023-01-avon-and-somerset-street.csv"

    DataO = cargar_datos(ruta)
    explorar_datos(DataO)
    Data = eliminar_columnas(DataO)
    explorar_datos(Data)
    Data = outliers(Data)
    Data = normalizar_texto(Data)
    print("Primeros registros del dataset limpio:")
    print(Data.head())
    grafico_dispersion_ultra(DataO, Data)
    return Data
    #Data.to_csv("Crimes_UK_Clean.csv", index=False)
    #print("Dataset limpio guardado como 'Crimes_UK_Clean.csv'")

cleanData()



