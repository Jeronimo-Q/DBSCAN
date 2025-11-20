from arbol import entrenar_arbol
from grafico import generateHeatmap, generateDiagramaBarras, generateHeatmapPlotly
from limpieza import cleanData
from DBSCAN import applyDBSCAN


if __name__ == "__main__":
    data=cleanData()
    generateHeatmapPlotly(data)
    generateHeatmap(data)
    generateDiagramaBarras(data)
    applyDBSCAN(data)
    entrenar_arbol(data)
    