"""Limpieza de datos, preprocesamiento y clasificadores del análisis de la campaña.

Reproduce el mismo flujo del notebook de la práctica para que el dashboard
muestre exactamente los mismos resultados.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, OrdinalEncoder,
                                   QuantileTransformer, StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from lvq import LVQ1

SEMILLA = 42
CATEGORICAS = ['Marital_Status']


# ----------------------------------------------------------------------------- datos
def cargar_datos(ruta='marketing_campaign.csv'):
    """
    Lee el archivo original del dataset.

    Argumentos:
        ruta (str): Ruta del archivo separado por tabuladores.

    Retorna:
        pandas.DataFrame: Datos sin modificar.
    """
    return pd.read_csv(ruta, sep='\t')


def limpiar(clientes):
    """
    Aplica la limpieza que no aprende de los datos.

    Quita columnas sin información y duplicados, crea Edad y Antiguedad, cambia los
    valores imposibles por NaN, ordena Education y aplica log(1 + x) al gasto y a
    las compras.

    Argumentos:
        clientes (pandas.DataFrame): Datos originales.

    Retorna:
        pandas.DataFrame: Datos limpios.
    """
    c = clientes.drop(columns=['ID', 'Z_CostContact', 'Z_Revenue']).drop_duplicates().copy()
    fecha = pd.to_datetime(c['Dt_Customer'], dayfirst=True)
    c['Antiguedad'] = (fecha.max() - fecha).dt.days
    c['Edad'] = fecha.max().year - c['Year_Birth']
    c = c.drop(columns=['Dt_Customer', 'Year_Birth'])
    c.loc[c['Edad'] > 100, 'Edad'] = np.nan
    c.loc[c['Income'] > 600000, 'Income'] = np.nan
    c['Marital_Status'] = c['Marital_Status'].replace({'Alone': 'Single', 'Absurd': np.nan, 'YOLO': np.nan})
    c['Education'] = c['Education'].map({'Basic': 0, 'Graduation': 1, 'Master': 2, '2n Cycle': 2, 'PhD': 3})
    sesgadas = [col for col in c.columns if col.startswith('Mnt') or col.startswith('Num')]
    c[sesgadas] = np.log1p(c[sesgadas])
    return c


def dividir(limpios):
    """
    Divide en entrenamiento (70 %) y prueba (30 %), estratificado.

    Argumentos:
        limpios (pandas.DataFrame): Datos limpios.

    Retorna:
        tuple: X_train, X_test, y_train, y_test.
    """
    X = limpios.drop('Response', axis=1)
    y = limpios['Response']
    return train_test_split(X, y, test_size=0.30, random_state=SEMILLA, stratify=y)


# ----------------------------------------------------------------------------- preprocesamiento
def crear_preprocesador(numericas, codificacion='onehot'):
    """
    Escala e imputa las numéricas con KNNImputer y codifica Marital_Status.

    Argumentos:
        numericas (list): Columnas numéricas.
        codificacion (str): 'onehot' o 'ordinal'.

    Retorna:
        sklearn.compose.ColumnTransformer: Preprocesador sin entrenar.
    """
    if codificacion == 'onehot':
        codificador = OneHotEncoder(drop='first', handle_unknown='ignore')
    else:
        codificador = OrdinalEncoder()
    return ColumnTransformer([
        ('num', Pipeline([('escalador', StandardScaler()), ('imputador', KNNImputer(n_neighbors=5))]), numericas),
        ('cat', Pipeline([('moda', SimpleImputer(strategy='most_frequent')), ('codificador', codificador)]), CATEGORICAS)
    ], verbose_feature_names_out=False)


def a_enteros(X, factor=1):
    """
    Multiplica los datos escalados por un factor y redondea a enteros.

    Argumentos:
        X (numpy.ndarray): Datos escalados (puede tener NaN).
        factor (float): Número por el que se multiplica antes de redondear.

    Retorna:
        numpy.ndarray: Datos enteros, los NaN se conservan.
    """
    return np.round(X * factor)


# ----------------------------------------------------------------------------- Gamma Pydra
TABLA_VARIANTES = {0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (1, 1, 0),
                   4: (0, 0, 1), 5: (1, 0, 1), 6: (0, 1, 1), 7: (1, 1, 1)}


def calcular_rho(X, criterio="min"):
    """
    Calcula la variable de paro rho: el minimo (o maximo) de los maximos de cada rasgo.

    Argumentos:
        X (numpy.ndarray): Datos de entrenamiento escalados a enteros (puede tener NaN).
        criterio (str): "min" o "max".

    Retorna:
        int: rho.
    """
    X = np.asarray(X, dtype=float)
    minimos = np.minimum(np.nanmin(X, axis=0), 0)
    maximos = np.nanmax(X - minimos, axis=0)
    rho = np.min(maximos) if criterio == "min" else np.max(maximos)
    return int(rho)


def operador_pydra(X_clase, y, theta, variante=0):
    """
    Compara rasgo por rasgo los patrones de una clase contra y.

    Si ambos valores existen da 1 cuando |x - y| <= theta; si falta alguno usa el
    valor de TABLA_VARIANTES.

    Argumentos:
        X_clase (numpy.ndarray): Patrones de una clase.
        y (numpy.ndarray): Patrón a clasificar.
        theta (int): Tolerancia.
        variante (int): Variante de Pydra (0 a 7).

    Retorna:
        numpy.ndarray: Matriz de 0 y 1.
    """
    X_clase = np.asarray(X_clase, dtype=float)
    y = np.asarray(y, dtype=float)
    v_xc, v_cx, v_xx = TABLA_VARIANTES[variante]
    falta_x = np.isnan(X_clase)
    falta_y = np.isnan(y)
    g = (np.abs(X_clase - y) <= theta).astype(int)
    g[falta_x & ~falta_y] = v_xc
    g[~falta_x & falta_y] = v_cx
    g[falta_x & falta_y] = v_xx
    return g


def gamma_pydra(df_clases, y, rho, variante=0):
    """
    Clasifica un patrón con la suma ponderada de cada clase; si hay empate incrementa theta.

    Argumentos:
        df_clases (list): Un DataFrame por clase.
        y (numpy.ndarray): Patrón a clasificar.
        rho (int): Variable de paro.
        variante (int): Variante de Pydra (0 a 7).

    Retorna:
        tuple: (índice de la clase, sumas ponderadas, theta final).
    """
    y = np.asarray(y, dtype=float)
    for theta in range(rho + 1):
        scores = np.array([operador_pydra(d.values, y, theta, variante).sum(axis=1).sum() / len(d)
                           for d in df_clases])
        ganadores = np.flatnonzero(scores == np.max(scores))
        if len(ganadores) == 1:
            return int(ganadores[0]), scores, theta
    return int(ganadores[0]), scores, rho


class GammaPydra(ClassifierMixin, BaseEstimator):
    """
    Clasificador Gamma Pydra con fit y predict para usarse en un Pipeline de sklearn.

    Argumentos:
        variante (int): Variante de Pydra (0 a 7).
        criterio_rho (str): "min" o "max" para calcular rho.
    """

    def __init__(self, variante=0, criterio_rho="min"):
        self.variante = variante
        self.criterio_rho = criterio_rho

    def fit(self, X, y):
        """Separa los patrones de entrenamiento por clase y calcula rho."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.df_clases_ = [pd.DataFrame(X[y == c]) for c in self.classes_]
        self.rho_ = calcular_rho(X, self.criterio_rho)
        return self

    def predict(self, X):
        """Clasifica cada patrón de X."""
        X = np.asarray(X, dtype=float)
        return np.array([self.classes_[gamma_pydra(self.df_clases_, fila, self.rho_, self.variante)[0]]
                         for fila in X])


# ----------------------------------------------------------------------------- modelos
def modelos(numericas):
    """
    Define cada clasificador con su preprocesamiento y los hiperparámetros a probar.

    Argumentos:
        numericas (list): Columnas numéricas.

    Retorna:
        dict: {nombre: (pipeline, param_grid)}.
    """
    smote = lambda: SMOTE(random_state=SEMILLA)
    normal = lambda: QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=SEMILLA)
    return {
        'SVM': (Pipeline([('preprocesador', crear_preprocesador(numericas)), ('svm', SVC(kernel='rbf'))]),
                {'svm__C': [0.1, 1, 10, 100], 'svm__gamma': ['scale', 0.01, 0.1],
                 'svm__class_weight': [None, 'balanced']}),
        'KNN': (ImbPipeline([('preprocesador', crear_preprocesador(numericas)), ('balanceo', smote()),
                             ('knn', KNeighborsClassifier())]),
                {'knn__n_neighbors': [3, 5, 7, 11, 15, 21], 'knn__weights': ['uniform', 'distance'],
                 'balanceo': [smote(), 'passthrough']}),
        'Logística': (Pipeline([('preprocesador', crear_preprocesador(numericas)),
                                ('logistica', LogisticRegression(solver='liblinear', random_state=SEMILLA))]),
                      {'logistica__C': [0.01, 0.1, 1, 10, 100], 'logistica__penalty': ['l1', 'l2'],
                       'logistica__class_weight': [None, 'balanced']}),
        'Árbol': (Pipeline([('preprocesador', crear_preprocesador(numericas, 'ordinal')),
                            ('arbol', DecisionTreeClassifier(random_state=SEMILLA))]),
                  {'arbol__max_depth': [3, 5, 7, 10, None], 'arbol__min_samples_leaf': [1, 5, 10, 20],
                   'arbol__criterion': ['gini', 'entropy'], 'arbol__class_weight': [None, 'balanced']}),
        'Random Forest': (Pipeline([('preprocesador', crear_preprocesador(numericas, 'ordinal')),
                                    ('bosque', RandomForestClassifier(random_state=SEMILLA))]),
                          {'bosque__n_estimators': [100, 300], 'bosque__max_depth': [5, 10, None],
                           'bosque__min_samples_leaf': [1, 5, 10], 'bosque__class_weight': [None, 'balanced']}),
        'Naive Bayes': (ImbPipeline([('preprocesador', crear_preprocesador(numericas)), ('normal', normal()),
                                     ('balanceo', smote()), ('bayes', GaussianNB())]),
                        {'normal': [normal(), 'passthrough'], 'bayes__var_smoothing': np.logspace(-9, 0, 10),
                         'balanceo': [smote(), 'passthrough']}),
        'Gamma Pydra': (Pipeline([('preprocesador', ColumnTransformer([('cat', OrdinalEncoder(), CATEGORICAS),
                                                                       ('num', 'passthrough', numericas)])),
                                  ('escalador', StandardScaler()),
                                  ('enteros', FunctionTransformer(a_enteros)),
                                  ('pydra', GammaPydra())]),
                        {'pydra__variante': list(range(8)),
                         'enteros__kw_args': [{'factor': f} for f in [0.5, 1, 2, 3, 5]]}),
        'LVQ': (ImbPipeline([('preprocesador', crear_preprocesador(numericas)), ('balanceo', smote()),
                             ('lvq', LVQ1(epocas=30, inicializacion='kmeans', random_state=SEMILLA))]),
                {'lvq__prototipos_por_clase': [1, 3, 5, 10], 'lvq__alpha0': [0.01, 0.05],
                 'balanceo': [smote(), 'passthrough']}),
    }


KFOLD = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
