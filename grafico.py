import matplotlib.pyplot as plt
import seaborn as sns

def generateGraffic(Data):
    Data['Crime type'].value_counts().plot(kind='bar', figsize=(10,5))
    plt.title("Frecuencia de tipos de crimen")
    plt.xlabel("Tipo de crimen")
    plt.ylabel("Número de casos")
    plt.show()


def generateHeatmap(Data):
    plt.figure(figsize=(8,6))
    sns.kdeplot(x=Data['Longitude'], y=Data['Latitude'], fill=True, cmap="Reds", thresh=0.05)
    plt.title("Mapa de calor de densidad de crímenes")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.show()

def graffic(Data):
    generateGraffic(Data)
    generateHeatmap(Data)