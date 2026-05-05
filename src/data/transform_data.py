import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

SCALER_PATH = "data/processed/scaler.pkl"

def main():
    if len(sys.argv) < 2:
        print("Uso: python src/data/transform_data.py <train.csv|test.csv>")
        sys.exit(1)

    filename = sys.argv[1]
    filepath = os.path.join("data/processed", filename)
    out_filename = filename.replace(".csv", "_transformed.csv")
    out_filepath = os.path.join("data/processed", out_filename)

    if not os.path.exists(filepath):
        print(f"❌ Error: No se encontró {filepath}")
        sys.exit(1)

    # Cargar los datos
    df = pd.read_csv(filepath)
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    # Lógica condicional según el archivo para evitar Data Leakage
    if filename == "train.csv":
        scaler = StandardScaler()
        # FIT + TRANSFORM solo en train
        X_scaled = scaler.fit_transform(X)
        
        # Guardar el escalador entrenado
        joblib.dump(scaler, SCALER_PATH)
        print("✅ Escalador ajustado en train y guardado correctamente.")

    elif filename == "test.csv":
        if not os.path.exists(SCALER_PATH):
            print("❌ Error: No se encontró scaler.pkl. Corre primero con train.csv")
            sys.exit(1)    
            scaler = joblib.load(SCALER_PATH)
    
        # SOLO TRANSFORM en test
        X_scaled = scaler.transform(X)
        print("✅ Escalador cargado y aplicado al test set.")
        
    else:
        print("❌ Archivo no reconocido. Usa train.csv o test.csv.")
        sys.exit(1)

    # Reconstruir y guardar
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    df_transformed = pd.concat([X_scaled_df, y], axis=1)
    
    df_transformed.to_csv(out_filepath, index=False)
    print(f"💾 Datos transformados guardados en {out_filepath}")

if __name__ == "__main__":
    main()