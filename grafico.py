import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
    plt.savefig("results/diagrama_barras_crimenes.png")
    plt.show()



def graficar_metricas_dbscan_arbol( y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print("accuracy:",accuracy)
    print("precision:",precision)
    print("recall:",recall)
    print("f1:",f1)

    metricas = [accuracy, precision, recall, f1]
    nombres = ["Accuracy", "Precision", "Recall", "F1"]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8,5))

    palette = sns.color_palette("coolwarm", len(metricas))

    ax = sns.barplot(x=nombres, y=metricas, palette=palette)

    for i, v in enumerate(metricas):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12, fontweight='bold')

    plt.ylim(0, 1.1)
    plt.title("Métricas del Árbol de Decisión", fontsize=16, fontweight='bold')
    plt.ylabel("Valor", fontsize=12)
    plt.xlabel("")

    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig("results/metricas_modelos.png", dpi=300)
    plt.show()