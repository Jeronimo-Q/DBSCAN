from arbol import entrenar_arbol
from grafico import generateHeatmap, generateDiagramaBarras , generateHeatmapPlotly, graficar_metricas_dbscan_arbol
from limpieza import cleanData
from DBSCAN import applyDBSCAN
from arbol2 import entrenar_arbol_decision


if __name__ == "__main__":
    data=cleanData()
    
    #generateDiagramaBarras(data)
    #generateHeatmap(data)
   # generateHeatmapPlotly(data)
    #applyDBSCAN(data)
    #entrenar_arbol(data)
    #entrenar_arbol_decision(data)
    #modelo, X_test, y_test, y_pred = entrenar_arbol(data)
    #graficar_metricas_dbscan_arbol(y_test,y_pred)
