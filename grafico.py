import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def generateHeatmap(Data):
    plt.figure(figsize=(8,6))
    sns.kdeplot(x=Data['Longitude'], y=Data['Latitude'], fill=True, cmap="Reds", thresh=0.05)
    plt.title("Mapa de calor de densidad de crímenes")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.savefig("results/heatmap_crimenes.png")
    plt.show()


def generateHeatmapPlotly(Data):

    fig = px.density_mapbox(
        Data,
        lat="Latitude",
        lon="Longitude",
        radius=10,                
        center=dict(lat=51.42, lon=-2.4),
        zoom=9,
        mapbox_style="open-street-map",  
        color_continuous_scale="Reds"
)

    fig.update_layout(
        title="Mapa de calor de crímenes en Londres"
    )

    fig.show()
    
def generateDiagramaBarras(data):
    crime_counts = data['Crime type'].value_counts().head(10)

    norm = plt.Normalize(crime_counts.min(), crime_counts.max())
    colors = plt.cm.viridis(norm(crime_counts.values))

    plt.figure(figsize=(10,6))
    sns.barplot(
        x=crime_counts.values,
        y=crime_counts.index,
        palette=colors
    )
    plt.title("Top 10 tipos de crímenes")
    plt.xlabel("Número de incidentes")
    plt.ylabel("Tipo de crimen")
    plt.show()
    plt.savefig("results/diagrama_barras_crimenes.png")