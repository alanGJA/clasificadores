"""Clasificadores de cuantización vectorial por aprendizaje (LVQ).

Implementación desde cero, compatible con scikit-learn (fit, predict, score,
Pipeline y validación cruzada), de tres variantes de la familia LVQ:

- LVQ1: regla original de Kohonen. Solo se mueve el prototipo ganador; se
  acerca a la muestra si comparten clase y se aleja si no.
- GLVQ: LVQ generalizado de Sato y Yamada. Minimiza un costo basado en la
  distancia relativa mu = (d+ - d-) / (d+ + d-).
- GMLVQ: GLVQ con métrica adaptable d(x, w) = (x - w)^T Omega^T Omega (x - w),
  de Schneider, Biehl y Hammer. La diagonal de Lambda = Omega^T Omega indica la
  relevancia de cada variable.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def inicializar_prototipos(X, y, clases, prototipos_por_clase, metodo, rng):
    """Coloca los prototipos iniciales de cada clase.

    Argumentos:
        X (numpy.ndarray): datos de entrenamiento, forma (n_muestras, n_variables).
        y (numpy.ndarray): etiqueta de cada muestra.
        clases (numpy.ndarray): etiquetas distintas, en orden.
        prototipos_por_clase (int): número de prototipos de cada clase.
        metodo (str): 'media' (centro de la clase), 'muestras' (muestras al azar
            de la clase) o 'kmeans' (centros de k-medias dentro de la clase).
        rng (numpy.random.Generator): generador de números aleatorios.

    Retorna:
        tuple: (W, c), con W de forma (n_prototipos, n_variables) y c el
            arreglo con la clase de cada prototipo.
    """
    W, c = [], []
    for clase in clases:
        Xc = X[y == clase]
        r = int(prototipos_por_clase)
        if metodo == "media":
            # con varios prototipos se agrega un poco de ruido para separarlos
            escala = 0.0 if r == 1 else 0.05 * Xc.std(axis=0)
            Wc = Xc.mean(axis=0) + rng.normal(size=(r, X.shape[1])) * escala
        elif metodo == "muestras":
            idx = rng.choice(len(Xc), size=r, replace=len(Xc) < r)
            Wc = Xc[idx]
        elif metodo == "kmeans":
            k = min(r, len(Xc))
            semilla = int(rng.integers(2**31 - 1))
            Wc = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(Xc).cluster_centers_
            if k < r:
                # si la clase tiene menos muestras que prototipos, se repiten centros
                Wc = Wc[rng.choice(k, size=r)]
        else:
            raise ValueError("inicializacion debe ser 'media', 'muestras' o 'kmeans'")
        W.append(np.asarray(Wc, dtype=float))
        c.extend([clase] * r)
    return np.vstack(W), np.asarray(c)


def distancias_cuadradas(X, W):
    """Distancia euclidiana al cuadrado entre cada muestra y cada prototipo.

    Argumentos:
        X (numpy.ndarray): muestras, forma (n_muestras, n_variables).
        W (numpy.ndarray): prototipos, forma (n_prototipos, n_variables).

    Retorna:
        numpy.ndarray: matriz de forma (n_muestras, n_prototipos).
    """
    d = (X**2).sum(axis=1)[:, None] - 2 * X @ W.T + (W**2).sum(axis=1)[None, :]
    return np.maximum(d, 0.0)


class LVQ1(BaseEstimator, ClassifierMixin):
    """Clasificador LVQ1 de Kohonen, entrenado muestra por muestra.

    Para cada muestra x de clase y se busca el prototipo más cercano w_k
    (el ganador). Si w_k es de la clase y se acerca a x; si no, se aleja:
    w_k <- w_k + s * alpha(t) * (x - w_k), con s = +1 o s = -1.

    Argumentos:
        prototipos_por_clase (int): prototipos que representan a cada clase.
        alpha0 (float): tasa de aprendizaje inicial, entre 0 y 1.
        epocas (int): pasadas completas por el conjunto de entrenamiento.
        inicializacion (str): 'muestras', 'media' o 'kmeans'.
        decaimiento (str): 'lineal' (alpha baja hasta cero) o 'constante'.
        guardar_historia (bool | int): con True guarda los prototipos al final de
            cada época en el atributo historia_; con un entero k los guarda cada
            k actualizaciones.
        prototipos_iniciales (numpy.ndarray | None): posiciones iniciales dadas a
            mano, ordenadas por clase; si se dan, se ignora inicializacion.
        random_state (int | None): semilla para inicializar y barajar.
    """

    def __init__(self, prototipos_por_clase=1, alpha0=0.05, epocas=30,
                 inicializacion="muestras", decaimiento="lineal",
                 guardar_historia=False, prototipos_iniciales=None, random_state=None):
        self.prototipos_por_clase = prototipos_por_clase
        self.alpha0 = alpha0
        self.epocas = epocas
        self.inicializacion = inicializacion
        self.decaimiento = decaimiento
        self.guardar_historia = guardar_historia
        self.prototipos_iniciales = prototipos_iniciales
        self.random_state = random_state

    def fit(self, X, y):
        """Entrena los prototipos con la regla LVQ1.

        Argumentos:
            X (numpy.ndarray): datos de entrenamiento, forma (n_muestras, n_variables).
            y (numpy.ndarray): etiquetas de clase.

        Retorna:
            LVQ1: el propio modelo ya entrenado.
        """
        X, y = check_X_y(X, y)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = unique_labels(y)
        if self.prototipos_iniciales is not None:
            W = np.array(self.prototipos_iniciales, dtype=float)
            c = np.repeat(self.classes_, self.prototipos_por_clase)
        else:
            W, c = inicializar_prototipos(X, y, self.classes_, self.prototipos_por_clase,
                                          self.inicializacion, rng)
        n = len(X)
        total = self.epocas * n
        paso = 0
        cada = self.guardar_historia if type(self.guardar_historia) is int else 0
        self.historia_ = [W.copy()] if self.guardar_historia else None
        for _ in range(self.epocas):
            for i in rng.permutation(n):                  # muestras en orden aleatorio
                if self.decaimiento == "lineal":
                    alpha = self.alpha0 * (1 - paso / total)  # tasa que baja hasta cero
                else:
                    alpha = self.alpha0
                dif = X[i] - W                            # x - w_k para cada prototipo
                k = np.argmin((dif ** 2).sum(axis=1))     # el ganador
                s = 1 if c[k] == y[i] else -1             # premio o castigo
                W[k] += s * alpha * dif[k]                # acercar o alejar
                paso += 1
                if cada and paso % cada == 0:
                    self.historia_.append(W.copy())
            if self.guardar_historia is True:
                self.historia_.append(W.copy())
        self.prototipos_ = W
        self.etiquetas_prototipos_ = c
        return self

    def predict(self, X):
        """Asigna a cada muestra la clase de su prototipo más cercano.

        Argumentos:
            X (numpy.ndarray): muestras, forma (n_muestras, n_variables).

        Retorna:
            numpy.ndarray: clase predicha para cada muestra.
        """
        check_is_fitted(self, "prototipos_")
        X = check_array(X)
        d = distancias_cuadradas(X, self.prototipos_)
        return self.etiquetas_prototipos_[np.argmin(d, axis=1)]


class GMLVQ(BaseEstimator, ClassifierMixin):
    """Familia GLVQ entrenada con L-BFGS: GLVQ, GRLVQ y GMLVQ en una sola clase.

    Minimiza el promedio de mu(x) = (d+ - d-) / (d+ + d-), donde d+ es la
    distancia al prototipo correcto más cercano y d- al prototipo incorrecto
    más cercano. mu < 0 quiere decir que la muestra quedó bien clasificada.
    La distancia es d(x, w) = (x - w)^T Omega^T Omega (x - w) y la forma de
    Omega define la variante:

    - metrica='euclidiana': Omega = I, el modelo es GLVQ.
    - metrica='diagonal': Omega diagonal, una relevancia por variable (GRLVQ).
    - metrica='matriz': Omega completa, también pesa combinaciones de
      variables (GMLVQ).

    Argumentos:
        prototipos_por_clase (int): prototipos que representan a cada clase.
        metrica (str): 'matriz', 'diagonal' o 'euclidiana'.
        beta (float | None): pendiente de la sigmoide Phi(mu); con None se usa
            Phi(mu) = mu.
        max_iter (int): iteraciones máximas del optimizador L-BFGS.
        inicializacion (str): 'media', 'muestras' o 'kmeans'.
        random_state (int | None): semilla.
    """

    def __init__(self, prototipos_por_clase=1, metrica="matriz", beta=None,
                 max_iter=2500, inicializacion="media", random_state=None):
        self.prototipos_por_clase = prototipos_por_clase
        self.metrica = metrica
        self.beta = beta
        self.max_iter = max_iter
        self.inicializacion = inicializacion
        self.random_state = random_state

    def _desempacar(self, theta, K, n):
        """Separa el vector de parámetros en prototipos y matriz Omega.

        Argumentos:
            theta (numpy.ndarray): parámetros aplanados.
            K (int): número de prototipos.
            n (int): número de variables.

        Retorna:
            tuple: (W, Omega).
        """
        W = theta[:K * n].reshape(K, n)
        if self.metrica == "matriz":
            Om = theta[K * n:].reshape(n, n)
        elif self.metrica == "diagonal":
            Om = np.diag(theta[K * n:])
        else:
            Om = np.eye(n)
        return W, Om

    def _costo_y_gradiente(self, theta, X, mismo, K):
        """Costo promedio Phi(mu) y su gradiente respecto a W y Omega.

        Argumentos:
            theta (numpy.ndarray): parámetros aplanados.
            X (numpy.ndarray): datos de entrenamiento.
            mismo (numpy.ndarray): máscara booleana (n_muestras, K), True si el
                prototipo es de la clase de la muestra.
            K (int): número de prototipos.

        Retorna:
            tuple: (costo, gradiente aplanado).
        """
        N, n = X.shape
        W, Om = self._desempacar(theta, K, n)
        D = X[:, None, :] - W[None, :, :]
        P = D @ Om.T
        d = np.einsum("ikm,ikm->ik", P, P)
        filas = np.arange(N)
        jmas = np.where(mismo, d, np.inf).argmin(axis=1)
        jmen = np.where(~mismo, d, np.inf).argmin(axis=1)
        dp, dm = d[filas, jmas], d[filas, jmen]
        s = dp + dm + 1e-12
        mu = (dp - dm) / s
        if self.beta is None:
            costo = mu.mean()
            dphi = np.ones(N)
        else:
            phi = 1.0 / (1.0 + np.exp(-self.beta * mu))
            costo = phi.mean()
            dphi = self.beta * phi * (1.0 - phi)
        # derivadas de mu respecto a d+ y d-
        gp = dphi * 2.0 * dm / s**2 / N
        gm = -dphi * 2.0 * dp / s**2 / N
        Pp, Pm = P[filas, jmas], P[filas, jmen]
        Dp, Dm = D[filas, jmas], D[filas, jmen]
        # dd/dw = -2 Omega^T Omega (x - w);  dd/dOmega = 2 Omega (x - w)(x - w)^T
        gW = np.zeros_like(W)
        np.add.at(gW, jmas, -2.0 * gp[:, None] * (Pp @ Om))
        np.add.at(gW, jmen, -2.0 * gm[:, None] * (Pm @ Om))
        grad = [gW.ravel()]
        if self.metrica in ("matriz", "diagonal"):
            gOm = 2.0 * ((gp[:, None] * Pp).T @ Dp + (gm[:, None] * Pm).T @ Dm)
            grad.append(gOm.ravel() if self.metrica == "matriz" else np.diag(gOm))
        return costo, np.concatenate(grad)

    def fit(self, X, y):
        """Entrena los prototipos y, si la métrica no es euclidiana, también Omega.

        Argumentos:
            X (numpy.ndarray): datos de entrenamiento, forma (n_muestras, n_variables).
            y (numpy.ndarray): etiquetas de clase.

        Retorna:
            GMLVQ: el propio modelo ya entrenado.
        """
        X, y = check_X_y(X, y)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = unique_labels(y)
        W0, c = inicializar_prototipos(X, y, self.classes_, self.prototipos_por_clase,
                                       self.inicializacion, rng)
        K, n = W0.shape
        mismo = y[:, None] == c[None, :]
        if self.metrica not in ("matriz", "diagonal", "euclidiana"):
            raise ValueError("metrica debe ser 'matriz', 'diagonal' o 'euclidiana'")
        theta0 = [W0.ravel()]
        if self.metrica == "matriz":
            theta0.append((np.eye(n) / np.sqrt(n)).ravel())
        elif self.metrica == "diagonal":
            theta0.append(np.ones(n) / np.sqrt(n))
        res = minimize(self._costo_y_gradiente, np.concatenate(theta0), jac=True,
                       args=(X, mismo, K), method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        W, Om = self._desempacar(res.x, K, n)
        # mu no cambia si Omega se multiplica por una constante: se normaliza traza(Lambda) = 1
        Om = Om / np.sqrt(np.trace(Om.T @ Om))
        self.prototipos_ = W
        self.etiquetas_prototipos_ = c
        self.omega_ = Om
        self.lambda_ = Om.T @ Om
        self.costo_ = res.fun
        self.iteraciones_ = res.nit
        return self

    def _distancias(self, X):
        """Distancia adaptable (x - w)^T Lambda (x - w) a cada prototipo.

        Argumentos:
            X (numpy.ndarray): muestras, forma (n_muestras, n_variables).

        Retorna:
            numpy.ndarray: matriz de forma (n_muestras, n_prototipos).
        """
        return distancias_cuadradas(X @ self.omega_.T, self.prototipos_ @ self.omega_.T)

    def predict(self, X):
        """Asigna a cada muestra la clase de su prototipo más cercano.

        Argumentos:
            X (numpy.ndarray): muestras, forma (n_muestras, n_variables).

        Retorna:
            numpy.ndarray: clase predicha para cada muestra.
        """
        check_is_fitted(self, "prototipos_")
        X = check_array(X)
        d = self._distancias(X)
        return self.etiquetas_prototipos_[np.argmin(d, axis=1)]

    def mu(self, X, y):
        """Distancia relativa mu de cada muestra; negativa si está bien clasificada.

        Argumentos:
            X (numpy.ndarray): muestras.
            y (numpy.ndarray): clase real de cada muestra.

        Retorna:
            numpy.ndarray: valores de mu entre -1 y 1.
        """
        d = self._distancias(check_array(X))
        mismo = np.asarray(y)[:, None] == self.etiquetas_prototipos_[None, :]
        dp = np.where(mismo, d, np.inf).min(axis=1)
        dm = np.where(~mismo, d, np.inf).min(axis=1)
        return (dp - dm) / (dp + dm)

    def relevancias(self):
        """Relevancia de cada variable: la diagonal de Lambda (suma 1).

        Retorna:
            numpy.ndarray: un valor por variable.
        """
        check_is_fitted(self, "lambda_")
        return np.diag(self.lambda_).copy()

    def proyectar(self, X, componentes=2):
        """Proyecta los datos sobre los vectores propios principales de Lambda.

        Argumentos:
            X (numpy.ndarray): muestras.
            componentes (int): número de ejes de la proyección.

        Retorna:
            numpy.ndarray: coordenadas, forma (n_muestras, componentes).
        """
        check_is_fitted(self, "lambda_")
        valores, vectores = np.linalg.eigh(self.lambda_)
        orden = np.argsort(valores)[::-1][:componentes]
        base = vectores[:, orden] * np.sqrt(np.maximum(valores[orden], 0.0))
        return np.asarray(X) @ base
