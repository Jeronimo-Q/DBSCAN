import matplotlib.pyplot as plt
import seaborn as sns


def generateHeatmap(Data):
    plt.figure(figsize=(8,6))
    sns.kdeplot(x=Data['Longitude'], y=Data['Latitude'], fill=True, cmap="Reds", thresh=0.05)
    plt.title("Mapa de calor de densidad de crímenes")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.show()
