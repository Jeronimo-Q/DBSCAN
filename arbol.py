import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def entrenar_arbol(data):

    le_lsoa = LabelEncoder()

    data["Month_num"] = data["Month"].str[-2:].astype(int)
    data["LSOA_code_encoded"] = le_lsoa.fit_transform(data["LSOA code"])

    data_filtrada = data[data['Cluster'] != -1]

    X = data_filtrada[['Latitude', 'Longitude','Month_num', 'LSOA_code_encoded']]
    y = data_filtrada['Cluster']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    modelo = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=5,
        random_state=42
    )

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Reporte
    print("\nREPORTE DEL MODELO\n")
    print(classification_report(y_test, y_pred))
    print("Precisión:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Árbol
    plt.figure(figsize=(18, 10))
    plot_tree(
        modelo,
        feature_names=['Latitude', 'Longitude','Month_num', 'LSOA code'],
        class_names=[str(c) for c in modelo.classes_],
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=3
    )
    plt.title("Árbol de Decisión - Clasificación de Zonas de Crimen")
    plt.savefig("results/arbol_decision.png")
    plt.show()

    return modelo, X_test, y_test, y_pred
