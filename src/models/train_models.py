import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

TRAIN_TRANSFORMED_PATH = "data/processed/train_transformed.csv"

def run_training():
    print("=== ENTRENAMIENTO Y EVALUACIÓN DE MODELOS (CV) ===")
    
    # 1. Cargar datos de entrenamiento escalados
    df_train = pd.read_csv(TRAIN_TRANSFORMED_PATH)
    X_train = df_train.drop(columns=['diagnosis'])
    y_train = df_train['diagnosis']
    
    # 2. Definir los modelos base
    models = {
        "Naive-Bayes": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "SVM": SVC(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42)
    }
    
    # 3. Definir los hiperparámetros a explorar (Punto 3.1 del TP)
    param_grids = {
        "Naive-Bayes": {}, # NB no tiene hiperparámetros críticos para ajustar acá
        "LDA": {},         # Idem LDA
        "SVM": {
            'C': [0.1, 0.5, 1, 10], 
            'kernel': ['linear', 'rbf']
        },
        "KNN": {
            'n_neighbors': [3, 5, 7, 11, 20], 
            'weights': ['uniform', 'distance']
        },
        "Random Forest": {
            'n_estimators': [50, 100, 200, 300], 
            'max_depth': [None, 5, 10, 15]
        }
    }
    
    results_detail = {}

    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Guardamos TODO: el mejor score, los parámetros y TODOS los resultados del grid
        results_detail[name] = {
            'best_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'all_scores': grid_search.cv_results_['mean_test_score'],
            'all_params': grid_search.cv_results_['params']
        }
        print(f"✅ {name} procesado.")

    return results_detail


if __name__ == "__main__":
    run_training()