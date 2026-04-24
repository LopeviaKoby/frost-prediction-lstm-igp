"""
Script de Ejemplo: Predicción Rápida de Heladas
Valle del Mantaro - IGP

Este script demuestra cómo usar los modelos entrenados
para hacer predicciones de heladas.
"""

import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

# Cargar modelos entrenados
print("📦 Cargando modelos...")
rf_model = joblib.load('../models/random_forest_model.pkl')
gb_model = joblib.load('../models/gradient_boosting_model.pkl')
mlp_model = keras.models.load_model('../models/mlp_model.h5')
scaler = joblib.load('../models/scaler.pkl')
feature_cols = joblib.load('../models/feature_columns.pkl')

print("✅ Modelos cargados exitosamente\n")

# Función de predicción
def predict_frost_ensemble(X, threshold=0.5):
    """
    Realiza predicciones usando ensamble de los 3 modelos.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Datos de entrada con las características requeridas
    threshold : float
        Umbral de probabilidad para clasificación
    
    Returns:
    --------
    dict
        Diccionario con predicciones de cada modelo y ensamble
    """
    # Asegurar que X tiene las columnas correctas
    X_selected = X[feature_cols]
    
    # Normalizar
    X_scaled = scaler.transform(X_selected)
    
    # Predicciones individuales
    rf_proba = rf_model.predict_proba(X_scaled)[:, 1]
    gb_proba = gb_model.predict_proba(X_scaled)[:, 1]
    mlp_proba = mlp_model.predict(X_scaled).flatten()
    
    # Ensamble (promedio de probabilidades)
    ensemble_proba = (rf_proba + gb_proba + mlp_proba) / 3
    
    # Clasificación
    results = {
        'rf_prediction': (rf_proba >= threshold).astype(int),
        'rf_probability': rf_proba,
        'gb_prediction': (gb_proba >= threshold).astype(int),
        'gb_probability': gb_proba,
        'mlp_prediction': (mlp_proba >= threshold).astype(int),
        'mlp_probability': mlp_proba,
        'ensemble_prediction': (ensemble_proba >= threshold).astype(int),
        'ensemble_probability': ensemble_proba
    }
    
    return results


# Ejemplo de uso
if __name__ == '__main__':
    print("="*70)
    print("EJEMPLO DE PREDICCIÓN DE HELADAS")
    print("="*70)
    
    # Cargar datos de ejemplo (últimas 100 observaciones del test set)
    print("\n📂 Cargando datos de ejemplo...")
    sample_data = pd.read_csv('../models/processed_data_sample.csv', index_col=0)
    sample_data.index = pd.to_datetime(sample_data.index)
    
    # Seleccionar algunas observaciones
    X_example = sample_data[feature_cols].head(5)
    
    print(f"✅ Datos cargados: {len(X_example)} observaciones\n")
    
    # Hacer predicciones
    print("🔮 Realizando predicciones...")
    predictions = predict_frost_ensemble(X_example, threshold=0.5)
    
    # Mostrar resultados
    print("\n" + "="*70)
    print("RESULTADOS DE PREDICCIÓN")
    print("="*70)
    
    results_df = pd.DataFrame({
        'Timestamp': X_example.index,
        'RF_Pred': predictions['rf_prediction'],
        'RF_Prob': predictions['rf_probability'].round(3),
        'GB_Pred': predictions['gb_prediction'],
        'GB_Prob': predictions['gb_probability'].round(3),
        'MLP_Pred': predictions['mlp_prediction'],
        'MLP_Prob': predictions['mlp_probability'].round(3),
        'Ensemble_Pred': predictions['ensemble_prediction'],
        'Ensemble_Prob': predictions['ensemble_probability'].round(3)
    })
    
    print(results_df.to_string(index=False))
    
    # Interpretación
    print("\n" + "="*70)
    print("INTERPRETACIÓN")
    print("="*70)
    print("0 = No Helada | 1 = Helada")
    print("\nConsideraciones:")
    print("- Ensemble combina las predicciones de los 3 modelos")
    print("- Probabilidad > 0.5 indica riesgo de helada")
    print("- Para alertas tempranas, se puede usar threshold < 0.5")
    print("- Gradient Boosting suele ser el más confiable")
    
    # Alertas
    print("\n" + "="*70)
    print("🚨 ALERTAS DE HELADA")
    print("="*70)
    
    high_risk = results_df[results_df['Ensemble_Prob'] >= 0.7]
    medium_risk = results_df[(results_df['Ensemble_Prob'] >= 0.4) & 
                             (results_df['Ensemble_Prob'] < 0.7)]
    
    if len(high_risk) > 0:
        print(f"\n⚠️ RIESGO ALTO ({len(high_risk)} casos):")
        print(high_risk[['Timestamp', 'Ensemble_Prob']].to_string(index=False))
    
    if len(medium_risk) > 0:
        print(f"\n⚡ RIESGO MEDIO ({len(medium_risk)} casos):")
        print(medium_risk[['Timestamp', 'Ensemble_Prob']].to_string(index=False))
    
    if len(high_risk) == 0 and len(medium_risk) == 0:
        print("\n✅ No se detectaron condiciones de riesgo significativo")
    
    print("\n" + "="*70)
    print("✅ Predicción completada")
    print("="*70)
