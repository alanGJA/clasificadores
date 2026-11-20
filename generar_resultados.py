"""Entrena los ocho clasificadores y guarda en resultados/ todo lo que muestra el dashboard.

Uso:
    python generar_resultados.py
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler

from pipeline import CATEGORICAS, KFOLD, cargar_datos, dividir, limpiar, modelos

SALIDA = 'resultados'


def metricas(y_real, y_pred):
    """
    Calcula las métricas de la clase Acepta.

    Argumentos:
        y_real (array): Clases reales.
        y_pred (array): Clases predichas.

    Retorna:
        dict: accuracy, precision, recall, f1, encontrados, perdidos y falsas alarmas.
    """
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    return {'accuracy': accuracy_score(y_real, y_pred),
            'precision': precision_score(y_real, y_pred, zero_division=0),
            'recall': recall_score(y_real, y_pred),
            'f1': f1_score(y_real, y_pred),
            'encontrados': int(((y_real == 1) & (y_pred == 1)).sum()),
            'perdidos': int(((y_real == 1) & (y_pred == 0)).sum()),
            'falsas_alarmas': int(((y_real == 0) & (y_pred == 1)).sum())}


def puntaje(modelo, X):
    """
    Regresa el puntaje continuo del modelo, o None si no lo tiene.

    Argumentos:
        modelo: Pipeline entrenado.
        X (pandas.DataFrame): Datos.

    Retorna:
        numpy.ndarray | None: decision_function o probabilidad de la clase 1.
    """
    if hasattr(modelo, 'decision_function'):
        return modelo.decision_function(X)
    if hasattr(modelo, 'predict_proba'):
        return modelo.predict_proba(X)[:, 1]
    return None


def comparar_imputacion(limpios):
    """
    Esconde 200 ingresos conocidos y compara la mediana contra KNNImputer.

    Argumentos:
        limpios (pandas.DataFrame): Datos limpios.

    Retorna:
        dict: Error absoluto medio de cada método y los valores reales y estimados.
    """
    numericas = [c for c in limpios.columns if c not in CATEGORICAS + ['Response']]
    conocidos = limpios[numericas].dropna()
    escalados = StandardScaler().fit_transform(conocidos)
    col = numericas.index('Income')
    ocultos = np.random.default_rng(42).choice(len(conocidos), size=200, replace=False)
    con_huecos = escalados.copy()
    con_huecos[ocultos, col] = np.nan
    media, desv = conocidos['Income'].mean(), conocidos['Income'].std()
    real = escalados[ocultos, col] * desv + media
    salida = {'real': real.round(0).tolist()}
    for nombre, imputador in [('Mediana', SimpleImputer(strategy='median')), ('KNNImputer', KNNImputer(n_neighbors=5))]:
        estimado = imputador.fit_transform(con_huecos)[ocultos, col] * desv + media
        salida[nombre] = {'mae': float(np.mean(np.abs(estimado - real))), 'estimado': estimado.round(0).tolist()}
    return salida


def main():
    os.makedirs(SALIDA, exist_ok=True)
    limpios = limpiar(cargar_datos())
    X_train, X_test, y_train, y_test = dividir(limpios)
    numericas = [c for c in X_train.columns if c not in CATEGORICAS]

    with open(f'{SALIDA}/imputacion.json', 'w') as f:
        json.dump(comparar_imputacion(limpios), f)

    filas, predicciones, mejores = [], {'y_test': y_test.values}, {}
    referencia = np.zeros(len(y_test), dtype=int)
    predicciones['pred_Referencia'] = referencia
    filas.append({'modelo': 'Referencia', 'f1_validacion': np.nan, 'auc': 0.5, **metricas(y_test, referencia),
                  'hiperparametros': 'Siempre predice "No acepta"'})

    for nombre, (pipe, grid_params) in modelos(numericas).items():
        print('Entrenando', nombre, flush=True)
        grid = GridSearchCV(pipe, grid_params, cv=KFOLD, scoring='f1', n_jobs=-1).fit(X_train, y_train)
        mejor = grid.best_estimator_
        mejores[nombre] = mejor
        pred = mejor.predict(X_test)
        score = puntaje(mejor, X_test)
        predicciones[f'pred_{nombre}'] = pred
        if score is not None:
            predicciones[f'score_{nombre}'] = score
        params = {k.split('__')[-1]: (('SMOTE' if str(v).startswith('SMOTE') else
                                       'QuantileTransformer' if str(v).startswith('Quantile') else
                                       'ninguno' if v == 'passthrough' else v) if not isinstance(v, dict) else v.get('factor'))
                  for k, v in grid.best_params_.items()}
        params = {('factor' if k == 'kw_args' else k): v for k, v in params.items()}
        filas.append({'modelo': nombre, 'f1_validacion': grid.best_score_,
                      'auc': roc_auc_score(y_test, score) if score is not None else np.nan,
                      **metricas(y_test, pred), 'hiperparametros': json.dumps(params, ensure_ascii=False, default=str)})

        res = pd.DataFrame(grid.cv_results_)
        cols = [c for c in res.columns if c.startswith('param_')]
        busqueda = pd.DataFrame({c.replace('param_', '').split('__')[-1]: res[c].apply(
            lambda v: 'SMOTE' if str(v).startswith('SMOTE') else 'QuantileTransformer' if str(v).startswith('Quantile')
            else 'ninguno' if v == 'passthrough' else v.get('factor') if isinstance(v, dict) else
            'None' if v is None else v) for c in cols})
        busqueda = busqueda.rename(columns={'kw_args': 'factor'})
        busqueda['f1_cv'] = res['mean_test_score']
        busqueda.to_csv(f'{SALIDA}/busqueda_{nombre}.csv', index=False)

    # Logística con umbral ajustado por validación cruzada
    log = mejores['Logística']
    proba_train = cross_val_predict(log, X_train, y_train, cv=KFOLD, method='predict_proba')[:, 1]
    pd.DataFrame({'y': y_train.values, 'proba': proba_train}).to_csv(f'{SALIDA}/logistica_cv.csv', index=False)
    precision, recall, umbrales = precision_recall_curve(y_train, proba_train)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    i = int(np.argmax(f1[:-1]))
    umbral = float(umbrales[i])
    proba_test = log.predict_proba(X_test)[:, 1]
    predicciones['proba_Logística'] = proba_test
    pred_u = (proba_test >= umbral).astype(int)
    predicciones['pred_Logística con umbral'] = pred_u
    predicciones['score_Logística con umbral'] = proba_test
    filas.append({'modelo': 'Logística con umbral', 'f1_validacion': float(f1[i]),
                  'auc': roc_auc_score(y_test, proba_test), **metricas(y_test, pred_u),
                  'hiperparametros': json.dumps({'umbral': round(umbral, 2)})})

    pd.DataFrame(filas).to_csv(f'{SALIDA}/metricas.csv', index=False)
    pd.DataFrame(predicciones).to_csv(f'{SALIDA}/predicciones.csv', index=False)

    rf = mejores['Random Forest']
    pd.Series(rf.named_steps['bosque'].feature_importances_,
              index=rf.named_steps['preprocesador'].get_feature_names_out(),
              name='importancia').sort_values(ascending=False).to_csv(f'{SALIDA}/importancias.csv')
    pd.Series(np.exp(log.named_steps['logistica'].coef_[0]),
              index=log.named_steps['preprocesador'].get_feature_names_out(),
              name='odds_ratio').sort_values().to_csv(f'{SALIDA}/odds.csv')
    print('Listo')


if __name__ == '__main__':
    main()
