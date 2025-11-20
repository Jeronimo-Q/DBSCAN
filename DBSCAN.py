from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def applyDBSCAN(Data, eps=0.01, min_samples=10):
    coords = Data[['Latitude', 'Longitude']].values

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    Data['Cluster'] = db.labels_

    # Métricas DBSCAN
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = list(db.labels_).count(-1)

    print(f"Clusters encontrados: {n_clusters}")
    print(f"Puntos de ruido: {n_noise}")
    print(Data['Cluster'].value_counts())

    # Gráfico DBSCAN
    plt.figure(figsize=(8,6))
    plt.scatter(Data['Longitude'], Data['Latitude'], c=Data['Cluster'], cmap='tab10', s=5)
    plt.title("Clusters detectados por DBSCAN")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.tight_layout()
    plt.savefig("results/dbscan_clusters.png")
    plt.show()

    return Data, n_clusters, n_noise
