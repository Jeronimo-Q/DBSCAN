from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def applyDBSCAN(Data, eps=0.01, min_samples=10):

    # Extrae Latitude y Longitude en formato numpy array (filas x 2)
    coords = Data[['Latitude', 'Longitude']].values

    # Crea y ajusta el modelo DBSCAN con los hiperparámetros dados
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    
    # Guarda las etiquetas resultantes en una nueva columna 'Cluster' (-1 indica ruido)
    Data['Cluster'] = db.labels_
    
    # Calcula el número de clusters encontrados (excluyendo el ruido -1)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    # Cuenta cuántos puntos fueron marcados como ruido
    n_noise = list(db.labels_).count(-1)
    
    # Imprime número de clusters y puntos de ruido
    print(f"Clusters encontrados: {n_clusters}")
    print(f"Puntos de ruido: {n_noise}")
    # Imprime la cantidad de registros por cluster (incluye -1 si existe)
    print(Data['Cluster'].value_counts())
    
    # Configura la figura para el scatter plot
    plt.figure(figsize=(8,6))
    # Dibuja los puntos coloreados por cluster; eje X = Longitud, eje Y = Latitud
    plt.scatter(Data['Longitude'], Data['Latitude'], c=Data['Cluster'], cmap='tab10', s=5)
    # Añade título y etiquetas a los ejes
    plt.title("Clusters detectados por DBSCAN")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    # Muestra la figura
    plt.show()
    
    # Devuelve el DataFrame con la columna 'Cluster'
    return Data
