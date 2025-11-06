# Importar librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def entrenar_arbol(data):

    le_lsoa = LabelEncoder()

    # 1️⃣ Eliminar los puntos de ruido que DBSCAN marcó con -1
    data["Month_num"] = data["Month"].str[-2:].astype(int)
    data["LSOA_code_encoded"] = le_lsoa.fit_transform(data["LSOA code"])

    data_filtrada = data[data['Cluster'] != -1]

    # 2️⃣ Seleccionar variables de entrada (Latitude y Longitude)
    X = data_filtrada[['Latitude', 'Longitude','Month_num', 'LSOA_code_encoded']]

    # 3️⃣ Seleccionar la variable objetivo (el cluster asignado por DBSCAN)
    y = data_filtrada['Cluster']

    # 4️⃣ Dividir los datos en entrenamiento (70%) y prueba (30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 5️⃣ Crear el modelo del árbol de decisión
    modelo = DecisionTreeClassifier(
        criterion='entropy',   # Usa entropía para medir pureza
        max_depth=5,           # Limita la profundidad del árbol (más simple e interpretable) evitando el sobreajuste
        random_state=42        # Fija semilla para reproducibilidad
    )

    # 6️⃣ Entrenar el modelo con los datos de entrenamiento
    modelo.fit(X_train, y_train)

    # 7️⃣ Realizar predicciones sobre el conjunto de prueba
    y_pred = modelo.predict(X_test)

    # 8️⃣ Evaluar el modelo con distintas métricas
    print("Reporte de Clasificación:\n")
    print(classification_report(y_test, y_pred))  # Precisión, Recall y F1 por clase
    print("Precisión del modelo:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))  # Compara clases reales vs predichas

    feature_importances = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": modelo.feature_importances_
    }).sort_values(by="Importancia", ascending=False)

    print ("\nImportancia de las variables:")
    print (feature_importances)

    # 9️⃣ Visualizar el árbol entrenado
    plt.figure(figsize=(18, 10))
    plot_tree(
        modelo,
        feature_names=['Latitude', 'Longitude','Month_num', 'LSOA code'],  # Variables usadas
        class_names=[str(c) for c in modelo.classes_],  # Nombres de clases (clusters)
        filled=True,       # Colorea nodos
        rounded=True,      # Bordes redondeados
        fontsize=10        # Tamaño de texto
    )
    plt.title("Árbol de Decisión - Clasificación de Zonas de Crimen")
    plt.show()

    # 🔟 Devolver el modelo entrenado para uso posterior
    return modelo
