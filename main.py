from arbol import entrenar_arbol
from grafico import generateHeatmap
from limpieza import cleanData
from DBSCAN import applyDBSCAN


if __name__ == "__main__":
    data=cleanData()
    generateHeatmap(data)
    applyDBSCAN(data)
    entrenar_arbol(data)