import matplotlib.pyplot as plt
import seaborn as sns


def generateHeatmap(Data):
    plt.figure(figsize=(8,6))
    sns.kdeplot(x=Data['Longitude'], y=Data['Latitude'], fill=True, cmap="Reds", thresh=0.05)
    plt.title("Mapa de calor de densidad de crímenes")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.show()
    
def generateDiagramaBarras(data):
    crime_counts = data['Crime type'].value_counts().head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=crime_counts.values, y=crime_counts.index, palette="viridis")
    plt.title("Top 10 tipos de crímenes")
    plt.xlabel("Número de incidentes")
    plt.ylabel("Tipo de crimen")
    plt.show()
