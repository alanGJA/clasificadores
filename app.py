"""Dashboard: ¿quién dirá que sí? Comparación de ocho clasificadores para una campaña de marketing.

Ejecutar:
    streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title='¿Quién dirá que sí?', page_icon='📬', layout='wide')

RAIZ = Path(__file__).parent
RES = RAIZ / 'resultados'

ACEPTA, NO_ACEPTA = '#2a78d6', '#b0aea5'
NAVY, NARANJA, TINTA = '#3b4589', '#f9ad30', '#1b1f3b'

st.markdown(f"""
<style>
  .block-container {{padding-top: 1.2rem; max-width: 1300px;}}
  .banda {{background: {NAVY}; border-radius: 14px; padding: 16px 26px; margin-bottom: 18px;}}
  .banda h1 {{color: {NARANJA}; margin: 0; font-size: 2rem;}}
  .titulo {{color: {NAVY}; font-weight: 700; font-size: 1.7rem; margin: 4px 0 6px 0;}}
  .idea {{color: {TINTA}; font-size: 1.15rem; margin: 0 0 8px 0;}}
  div[data-testid="stMetricValue"] {{font-size: 2.3rem;}}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- datos
@st.cache_data
def cargar():
    """Lee el dataset original y los resultados que guardó generar_resultados.py."""
    d = pd.read_csv(RAIZ / 'marketing_campaign.csv', sep='\t')
    d['Campañas previas'] = d[[f'AcceptedCmp{i}' for i in range(1, 6)]].sum(axis=1)
    met = pd.read_csv(RES / 'metricas.csv').set_index('modelo')
    pred = pd.read_csv(RES / 'predicciones.csv')
    busq = {p.stem.replace('busqueda_', ''): pd.read_csv(p) for p in RES.glob('busqueda_*.csv')}
    imp = pd.read_csv(RES / 'importancias.csv', index_col=0)['importancia']
    imput = json.loads((RES / 'imputacion.json').read_text())
    return d, met, pred, busq, imp, imput


crudos, met, pred, busq, imp, imput = cargar()
y_test = pred['y_test'].values


def estilo(fig, alto=360, titulo=None, leyenda=False):
    """Estilo común de las gráficas."""
    fig.update_layout(height=alto, title=titulo, margin=dict(l=10, r=10, t=50, b=60 if leyenda else 10),
                      plot_bgcolor='#fcfcfb', paper_bgcolor='#fcfcfb', font=dict(size=15, color=TINTA),
                      showlegend=leyenda, legend=dict(orientation='h', yanchor='top', y=-0.25, x=0),
                      hoverlabel=dict(font_size=14), title_font=dict(size=18, color=TINTA))
    fig.update_xaxes(showgrid=False, linecolor='#c3c2b7', tickfont_size=14)
    fig.update_yaxes(gridcolor='#ebeae6', zeroline=False, tickfont_size=14)
    return fig


def mostrar(fig):
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


def barras_h(etiquetas, valores, colores, texto, titulo, alto=360, eje=None):
    """Barras horizontales con el valor al final de cada barra."""
    fig = go.Figure(go.Bar(y=etiquetas, x=valores, orientation='h', marker_color=colores, marker_cornerradius=4,
                           text=texto, textposition='outside', hovertemplate='%{y}: %{x:,.2f}<extra></extra>'))
    rango = [min(0, min(valores) * 1.3), max(valores) * 1.25]
    fig.update_xaxes(range=rango, title=eje)
    return estilo(fig, alto, titulo)


# ----------------------------------------------------------------------------- navegación
st.markdown('<div class="banda"><h1>¿Quién dirá que sí?</h1></div>', unsafe_allow_html=True)
SECCIONES = ['1 · Los datos', '2 · Preprocesamiento', '3 · Los clasificadores', '4 · Resultados']
with st.sidebar:
    seccion = st.radio('Recorrido', SECCIONES)
    st.divider()
    st.caption('Analítica Avanzada de Datos · 6AM1  \nEquipo 3')


# ----------------------------------------------------------------------------- 1. datos
if seccion == SECCIONES[0]:
    st.markdown('<div class="titulo">Solo 1 de cada 7 clientes acepta</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    k1.metric('Clientes', f'{len(crudos):,}')
    k2.metric('Aceptaron la oferta', f'{crudos.Response.sum()} ({crudos.Response.mean():.0%})')
    k3.metric('Accuracy de decir «nadie acepta»', f'{1 - crudos.Response.mean():.0%}')

    c1, c2 = st.columns(2)
    with c1:
        conteo = crudos['Response'].value_counts().sort_index()
        mostrar(barras_h(['No acepta', 'Acepta'], conteo.values, [NO_ACEPTA, ACEPTA],
                         [f'{v:,}' for v in conteo.values], 'Balance de clases', 300))
    with c2:
        tasa = crudos.groupby('Campañas previas')['Response'].mean()
        fig = go.Figure(go.Bar(x=tasa.index.astype(str), y=tasa.values, marker_color=ACEPTA, marker_cornerradius=4,
                               text=[f'{v:.0%}' for v in tasa.values], textposition='outside',
                               hovertemplate='%{x} campañas previas: %{y:.0%} acepta<extra></extra>'))
        fig.update_yaxes(tickformat='.0%', range=[0, 1.1])
        fig.update_xaxes(title='Campañas anteriores aceptadas')
        mostrar(estilo(fig, 300, 'Quien ya aceptó antes, vuelve a aceptar'))


# ----------------------------------------------------------------------------- 2. preprocesamiento
elif seccion == SECCIONES[1]:
    st.markdown('<div class="titulo">Cada clasificador recibe los datos como los necesita</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.6])
    with c1:
        mae = [imput['Mediana']['mae'], imput['KNNImputer']['mae']]
        mostrar(barras_h(['Mediana', 'KNNImputer'], mae, [NO_ACEPTA, ACEPTA], [f'{v:,.0f}' for v in mae],
                         'Error al estimar 200 ingresos ocultos', 260))
        st.markdown('<p class="idea">KNNImputer estima con los 5 clientes más parecidos: '
                    '<b>menos de la mitad del error</b>.</p>', unsafe_allow_html=True)
    with c2:
        tabla = pd.DataFrame({
            'Clasificador': ['SVM', 'KNN', 'Logística', 'Árbol', 'Random Forest', 'Naive Bayes', 'Gamma Pydra', 'LVQ'],
            'Valores perdidos': ['KNNImputer'] * 6 + ['Sin imputar', 'KNNImputer'],
            'Escala': ['Estandarizar', 'Estandarizar', 'Estandarizar', '—', '—', 'Estandarizar',
                       'z-score a enteros', 'Estandarizar'],
            'Desbalance': ['class_weight', 'SMOTE', 'class_weight', 'class_weight', 'class_weight', 'SMOTE',
                           'Suma ponderada', 'SMOTE'],
        })
        st.dataframe(tabla, hide_index=True, width='stretch', height=330)
    st.markdown('<p class="idea">Antes: se quitaron 182 duplicados, los valores imposibles pasaron a NaN '
                '(edades de 120 años, ingreso de 666,666) y el gasto se transformó con log(1+x).</p>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------- 3. clasificadores
elif seccion == SECCIONES[2]:
    idea = {
        'SVM': 'Busca la frontera con el mayor margen entre las clases.',
        'KNN': 'Vota entre los k clientes más parecidos.',
        'Logística': 'Calcula la probabilidad de aceptar.',
        'Árbol': 'Hace preguntas sobre una variable a la vez.',
        'Random Forest': 'Muchos árboles que votan.',
        'Naive Bayes': 'Probabilidad suponiendo variables normales.',
        'Gamma Pydra': 'Cuenta coincidencias rasgo por rasgo, sin imputar.',
        'LVQ': 'Resume cada clase en unos cuantos prototipos.',
    }
    nombre = st.segmented_control('Clasificador', list(idea), default='SVM', label_visibility='collapsed') or 'SVM'
    fila = met.loc[nombre]
    st.markdown(f'<div class="titulo">{nombre}</div><p class="idea">{idea[nombre]}</p>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('f1', f'{fila.f1:.2f}')
    m2.metric('Encontrados', f'{int(fila.encontrados)} de 94')
    m3.metric('Falsas alarmas', int(fila.falsas_alarmas))
    m4.metric('Precision', f'{fila.precision:.2f}')

    c1, c2 = st.columns(2)
    with c1:
        cm = confusion_matrix(y_test, pred[f'pred_{nombre}'])
        etiquetas = [['Bien descartados', 'Falsas alarmas'], ['Perdidos', 'Encontrados']]
        fig = go.Figure(go.Heatmap(z=cm, x=['Predice: no', 'Predice: sí'], y=['Real: no', 'Real: sí'],
                                   text=[[f'{etiquetas[i][j]}<br><b>{cm[i, j]}</b>' for j in range(2)] for i in range(2)],
                                   texttemplate='%{text}', textfont=dict(size=17), showscale=False,
                                   colorscale=[[0, '#eef3fb'], [1, ACEPTA]], hoverinfo='skip'))
        fig.update_yaxes(autorange='reversed')
        mostrar(estilo(fig, 340, 'Matriz de confusión'))
    with c2:
        b = busq[nombre].copy()
        if 'class_weight' in b:
            b['class_weight'] = b['class_weight'].fillna('sin balancear').replace('balanced', 'balanceado')
        if 'balanceo' in b:
            b['balanceo'] = b['balanceo'].replace({'ninguno': 'sin balancear'})

        def lineas(x, serie, titulo, etiqueta_x):
            t = b.pivot_table(index=x, columns=serie, values='f1_cv', aggfunc='max')
            fig = go.Figure()
            for col, color in zip(t.columns, [ACEPTA, '#eb6834']):
                fig.add_trace(go.Scatter(x=t.index.astype(str), y=t[col], mode='lines+markers', name=str(col),
                                         line=dict(width=2, color=color), marker=dict(size=9),
                                         hovertemplate='%{x}: f1 %{y:.2f}<extra></extra>'))
            fig.update_xaxes(title=etiqueta_x, type='category')
            fig.update_yaxes(title='f1 (validación)')
            return estilo(fig, 340, titulo, leyenda=True)

        if nombre == 'SVM':
            mostrar(lineas('C', 'class_weight', 'Balancear las clases sube el f1', 'C'))
        elif nombre == 'KNN':
            mostrar(lineas('n_neighbors', 'balanceo', 'Sin SMOTE, más vecinos = peor', 'Vecinos'))
        elif nombre == 'Logística':
            u = st.slider('Umbral para marcar «acepta»', 0.1, 0.9, 0.5, 0.05)
            proba = pred['proba_Logística'].values
            p = (proba >= u).astype(int)
            tp, fp = int(((y_test == 1) & (p == 1)).sum()), int(((y_test == 0) & (p == 1)).sum())
            prec, rec = tp / max(tp + fp, 1), tp / 94
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            fig = go.Figure(go.Bar(x=['Encontrados', 'Falsas alarmas'], y=[tp, fp], marker_color=[ACEPTA, NO_ACEPTA],
                                   marker_cornerradius=4, text=[tp, fp], textposition='outside'))
            fig.update_yaxes(range=[0, 200])
            mostrar(estilo(fig, 270, f'Umbral {u:.2f} · f1 = {f1:.2f}'))
            st.caption('El mejor umbral, elegido con validación cruzada, es 0.70: f1 de 0.68.')
        elif nombre == 'Árbol':
            mostrar(lineas('max_depth', 'class_weight', 'Profundidad del árbol', 'Profundidad máxima'))
        elif nombre == 'Random Forest':
            top = imp.head(6)[::-1]
            mostrar(barras_h(list(top.index), list(top.values), ACEPTA, [f'{v:.2f}' for v in top.values],
                             'Variables más importantes', 340))
        elif nombre == 'Naive Bayes':
            b['var_smoothing'] = b['var_smoothing'].map(lambda v: f'{v:.0e}')
            mostrar(lineas('var_smoothing', 'balanceo', 'SMOTE mejora a Naive Bayes', 'var_smoothing'))
        elif nombre == 'Gamma Pydra':
            t = b.groupby('factor')['f1_cv'].max()
            fig = go.Figure(go.Bar(x=t.index.astype(str), y=t.values, marker_color=ACEPTA, marker_cornerradius=4,
                                   text=[f'{v:.2f}' for v in t.values], textposition='outside'))
            fig.update_xaxes(title='Factor del z-score', type='category')
            fig.update_yaxes(title='f1 (validación)', range=[0, 0.55])
            mostrar(estilo(fig, 340, 'El factor importa; la variante casi no'))
        elif nombre == 'LVQ':
            mostrar(lineas('prototipos_por_clase', 'balanceo', 'Más prototipos, mejor', 'Prototipos por clase'))


# ----------------------------------------------------------------------------- 4. resultados
elif seccion == SECCIONES[3]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="titulo">¿Quién ganó?</div>', unsafe_allow_html=True)
        m = met.drop('Referencia').sort_values('f1')
        mejor = m.index[-1]
        mostrar(barras_h(list(m.index), list(m.f1), [NARANJA if n == mejor else ACEPTA for n in m.index],
                         [f'{v:.2f}' for v in m.f1], 'f1 en prueba', 470))
    with c2:
        st.markdown('<div class="titulo">¿Le conviene a la tienda?</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        costo = a.slider('Costo por contacto', 1.0, 10.0, 3.0, 0.5)
        ingreso = b.slider('Ingreso por cliente que acepta', 5.0, 30.0, 11.0, 0.5)
        estrategias = {'Mandar a todos': (94, 524), 'SVM': None, 'Logística con umbral': None, 'LVQ': None}
        gan = {}
        for e, v in estrategias.items():
            tp, fp = v or (int(met.loc[e, 'encontrados']), int(met.loc[e, 'falsas_alarmas']))
            gan[e] = ingreso * tp - costo * (tp + fp)
        g = pd.Series(gan).sort_values()
        mostrar(barras_h(list(g.index), list(g.values),
                         [NO_ACEPTA if e == 'Mandar a todos' else (NARANJA if v == g.max() else ACEPTA)
                          for e, v in g.items()],
                         [f'{v:,.0f}' for v in g.values], 'Ganancia con los 618 clientes de prueba', 320))
        st.markdown('<p class="idea">Mandar la oferta a todos <b>pierde dinero</b>; '
                    'dirigirla con el modelo la vuelve <b>rentable</b>.</p>', unsafe_allow_html=True)
