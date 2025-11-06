from sklearn.model_selection import train_test_split             
from sklearn.tree import DecisionTreeClassifier                 
from sklearn.preprocessing import LabelEncoder                   
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  
from sklearn import tree                                     
import matplotlib.pyplot as plt                               
import pandas as pd                                               

# Definición de la función principal
def entrenar_arbol_decision(df):

    # Convierte la columna 'Month' (por ejemplo '2025-05') en un número (ejemplo: 5)
    # Esto permite al modelo entender la variable temporal en formato numérico.
    df["Month_num"] = df["Month"].str[-2:].astype(int)

    # Creación de codificadores para las variables categóricas
    le_lsoa = LabelEncoder()   
    le_crime = LabelEncoder() 

    # Aplicar la codificación sobre las columnas de texto
    df["LSOA_code_encoded"] = le_lsoa.fit_transform(df["LSOA code"])
    df["Crime_type_encoded"] = le_crime.fit_transform(df["Crime type"])

    # Definición de las variables independientes (características)
    # Incluyen ubicación, mes y área
    X = df[["Longitude", "Latitude", "Month_num", "LSOA_code_encoded"]]

    # Definición de la variable objetivo (lo que se desea predecir)
    y = df["Crime_type_encoded"]

    # División del conjunto de datos en entrenamiento (70%) y prueba (30%)
    # Esto permite evaluar el rendimiento del modelo con datos no vistos.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Creación del modelo de Árbol de Decisión
    model = DecisionTreeClassifier(
        criterion="entropy",  # Utiliza la entropía como medida de impureza
        max_depth=8,          # Limita la profundidad del árbol para evitar sobreajuste
        random_state=42       # Asegura resultados reproducibles
    )

    # Entrenamiento del modelo con los datos de entrenamiento
    model.fit(X_train, y_train)

    # Realización de predicciones sobre el conjunto de prueba
    y_pred = model.predict(X_test)

    # Cálculo del nivel de precisión del modelo (porcentaje de aciertos)
    acc = accuracy_score(y_test, y_pred)
    print("Precisión (accuracy):", round(acc, 4))

    # Mostrar la matriz de confusión (compara clases reales y predichas)
    print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))

    # Mostrar el reporte de clasificación
    # Incluye precisión, recall y F1-score para cada tipo de crimen
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, target_names=le_crime.classes_))

    # Calcular la importancia de cada variable utilizada en el modelo
    feature_importances = pd.DataFrame({
        "Variable": X.columns,
        "Importancia": model.feature_importances_
    }).sort_values(by="Importancia", ascending=False)

    # Mostrar las variables más relevantes para la predicción
    print("\nImportancia de las variables:")
    print(feature_importances)

    # Visualización del árbol de decisión
    plt.figure(figsize=(16,8))
    tree.plot_tree(
        model,
        feature_names=X.columns,           # Nombres de las variables predictoras
        class_names=le_crime.classes_,     # Nombres de las clases objetivo
        filled=True,                       # Colorea los nodos según la clase
        rounded=True,                      # Bordes redondeados para mejor visualización
        fontsize=9,                        # Tamaño del texto
        max_depth=3                        # Muestra solo las primeras capas del árbol
    )
    plt.show()

    # Devuelve el modelo entrenado y las importancias de las variables
    return model, feature_importances
