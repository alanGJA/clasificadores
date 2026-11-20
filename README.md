# ¿Quién dirá que sí? · Dashboard

Dashboard de la comparación de ocho clasificadores (SVM, KNN, Regresión Logística, Árbol de decisión,
Random Forest, Naive Bayes, Gamma Pydra y LVQ) para predecir qué clientes aceptan una campaña de marketing.
Dataset: [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis).

Analítica Avanzada de Datos · 6AM1 · Equipo 3: Sánchez Oliver Cristóbal, García Juanillo Alan, Reyes León Ángel Eduardo.

## Archivos

| Archivo | Qué hace |
|:--|:--|
| `app.py` | El dashboard de Streamlit |
| `pipeline.py` | Limpieza, preprocesamiento, Gamma Pydra y los ocho modelos con sus hiperparámetros |
| `generar_resultados.py` | Entrena todo y guarda las métricas y predicciones en `resultados/` |
| `lvq.py` | Implementación de LVQ del equipo |
| `marketing_campaign.csv` | Datos originales |

## Correrlo en la computadora

```bash
pip install -r requirements.txt
python generar_resultados.py   # solo si se cambia algo del modelo (tarda ~1 min)
streamlit run app.py
```

## Publicarlo en Streamlit Community Cloud

1. Subir esta carpeta a un repositorio de GitHub (público), con `resultados/` incluida.
2. Entrar a https://share.streamlit.io, iniciar sesión con GitHub y elegir **Create app**.
3. Elegir el repositorio, la rama `main` y el archivo `app.py`. Publicar.

## Reparto de la exposición (15 minutos)

| Parte | Quién | Tiempo |
|:--|:--|:--|
| Canva 1–6: la historia, el problema y los datos | Cristóbal | 2:00 |
| Canva 7–10: la trampa de la accuracy, los clasificadores y el plan | Alan | 1:20 |
| Dashboard 1 · Los datos | Alan | 1:30 |
| Dashboard 2 · Preprocesamiento | Alan | 1:30 |
| Dashboard 3 · Los clasificadores | Ángel | 4:00 |
| Dashboard 4 · Resultados: ¿quién ganó? | Ángel | 1:00 |
| Dashboard 4 · Resultados: ¿le conviene a la tienda? y cierre | Cristóbal | 1:30 |
| **Total** | | **12:50** (2 min de margen) |
