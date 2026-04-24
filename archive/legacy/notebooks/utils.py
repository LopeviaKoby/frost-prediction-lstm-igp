"""
Utilidades para el Proyecto de Predicción de Heladas
Valle del Mantaro - IGP

Este módulo contiene funciones auxiliares para:
- Carga de datos
- Preprocesamiento
- Evaluación de modelos
- Visualizaciones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
import joblib


def load_ema_datasets(data_path='../data/'):
    """
    Carga e integra los 7 datasets de estaciones EMA.
    
    Parameters:
    -----------
    data_path : str
        Ruta al directorio con los archivos CSV
    
    Returns:
    --------
    pd.DataFrame
        DataFrame integrado con todas las variables
    """
    datasets = {
        'tempsup': 'tempsup_hourly_2018_2025.csv',
        'HR': 'HR_hourly_2018_2025.csv',
        'radinf': 'radinf_hourly_2018_2025.csv',
        'dir': 'dir_hourly_2018_2025.csv',
        'vel': 'vel_hourly_2018_2025.csv',
        'pp': 'pp_hourly_2018_2025.csv',
        'press': 'press_hourly_2018_2025.csv'
    }
    
    dataframes = {}
    for var_name, file_name in datasets.items():
        df = pd.read_csv(data_path + file_name, index_col=0)
        df.columns = [f'{var_name}_{stat}' for stat in ['mean', 'std', 'max', 'min']]
        df.index = pd.to_datetime(df.index)
        df.index.name = 'timestamp'
        dataframes[var_name] = df
    
    df_integrated = pd.concat(dataframes.values(), axis=1)
    return df_integrated


def create_frost_target(df, temp_col='tempsup_min', threshold=0):
    """
    Crea la variable objetivo de helada.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con datos climáticos
    temp_col : str
        Nombre de la columna de temperatura mínima
    threshold : float
        Umbral de temperatura para definir helada (default: 0°C)
    
    Returns:
    --------
    pd.Series
        Serie binaria con eventos de helada (1) o no (0)
    """
    return (df[temp_col] <= threshold).astype(int)


def add_temporal_features(df):
    """
    Agrega características temporales al DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con índice temporal
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con características temporales añadidas
    """
    df = df.copy()
    
    # Básicas
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['quarter'] = df.index.quarter
    
    # Cíclicas
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Binarias
    df['is_winter'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_night'] = df['hour'].isin(range(0, 7)).astype(int)
    
    return df


def add_lag_features(df, variables, lags=[1, 3, 6, 12]):
    """
    Agrega características de rezago (lag) temporal.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con series temporales
    variables : list
        Lista de nombres de variables para crear lags
    lags : list
        Lista de rezagos en horas
    
    Returns:
    --------
    pd.DataFrame
        DataFrame con características de lag añadidas
    """
    df = df.copy()
    
    for var in variables:
        for lag in lags:
            df[f'{var}_lag{lag}h'] = df[var].shift(lag)
    
    return df


def evaluate_model(y_true, y_pred, y_proba=None, model_name='Model'):
    """
    Evalúa un modelo de clasificación y muestra métricas.
    
    Parameters:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_pred : array-like
        Predicciones del modelo
    y_proba : array-like, optional
        Probabilidades predichas
    model_name : str
        Nombre del modelo para el reporte
    
    Returns:
    --------
    dict
        Diccionario con métricas de evaluación
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    print(f"\n{'='*60}")
    print(f"EVALUACIÓN: {model_name}")
    print(f"{'='*60}")
    for metric, value in metrics.items():
        if metric != 'model':
            print(f"{metric.upper():15s}: {value:.4f}")
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    Visualiza la matriz de confusión.
    
    Parameters:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_pred : array-like
        Predicciones del modelo
    title : str
        Título del gráfico
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Helada', 'Helada'],
                yticklabels=['No Helada', 'Helada'])
    plt.xlabel('Predicción', fontsize=12, fontweight='bold')
    plt.ylabel('Real', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_proba, model_name='Model'):
    """
    Visualiza la curva ROC.
    
    Parameters:
    -----------
    y_true : array-like
        Etiquetas verdaderas
    y_proba : array-like
        Probabilidades predichas
    model_name : str
        Nombre del modelo
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_model(model, filepath, model_type='sklearn'):
    """
    Guarda un modelo entrenado.
    
    Parameters:
    -----------
    model : object
        Modelo entrenado
    filepath : str
        Ruta donde guardar el modelo
    model_type : str
        Tipo de modelo ('sklearn' o 'keras')
    """
    if model_type == 'sklearn':
        joblib.dump(model, filepath)
    elif model_type == 'keras':
        model.save(filepath)
    
    print(f"✅ Modelo guardado en: {filepath}")


def load_model(filepath, model_type='sklearn'):
    """
    Carga un modelo guardado.
    
    Parameters:
    -----------
    filepath : str
        Ruta del modelo guardado
    model_type : str
        Tipo de modelo ('sklearn' o 'keras')
    
    Returns:
    --------
    object
        Modelo cargado
    """
    if model_type == 'sklearn':
        model = joblib.load(filepath)
    elif model_type == 'keras':
        from tensorflow import keras
        model = keras.models.load_model(filepath)
    
    print(f"✅ Modelo cargado desde: {filepath}")
    return model


def predict_frost(model, X, scaler=None, threshold=0.5):
    """
    Realiza predicciones de heladas.
    
    Parameters:
    -----------
    model : object
        Modelo entrenado
    X : pd.DataFrame or np.ndarray
        Características de entrada
    scaler : object, optional
        Scaler para normalización
    threshold : float
        Umbral de probabilidad para clasificación
    
    Returns:
    --------
    tuple
        (predicciones binarias, probabilidades)
    """
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    # Obtener probabilidades
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_scaled)[:, 1]
    else:
        proba = model.predict(X_scaled).flatten()
    
    # Clasificar según umbral
    predictions = (proba >= threshold).astype(int)
    
    return predictions, proba


if __name__ == '__main__':
    print("Módulo de utilidades cargado correctamente")
    print("Funciones disponibles:")
    print("  - load_ema_datasets()")
    print("  - create_frost_target()")
    print("  - add_temporal_features()")
    print("  - add_lag_features()")
    print("  - evaluate_model()")
    print("  - plot_confusion_matrix()")
    print("  - plot_roc_curve()")
    print("  - save_model()")
    print("  - load_model()")
    print("  - predict_frost()")
