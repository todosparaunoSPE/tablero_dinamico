# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 09:50:25 2025

@author: jperezr
"""


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# Configuración de la página
st.set_page_config(page_title="Tablero Multiusuario", layout="wide")

# Estilo de fondo
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background:
radial-gradient(black 15%, transparent 16%) 0 0,
radial-gradient(black 15%, transparent 16%) 8px 8px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 0 1px,
radial-gradient(rgba(255,255,255,.1) 15%, transparent 20%) 8px 9px;
background-color:#282828;
background-size:16px 16px;
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# Función para generar datos dinámicos para cada usuario
def generate_user_data(role):
    if role == "Director":
        return {
            "Producción Diaria": np.random.randint(100, 200),
            "Ingresos Totales": np.random.randint(500, 1000),
            "Rentabilidad (%)": np.random.randint(20, 50),
            "Eficiencia Operativa (%)": np.random.randint(80, 95),
        }
    elif role == "Analista":
        return {
            "Análisis Completados": np.random.randint(10, 30),
            "Errores Detectados": np.random.randint(0, 10),
            "Tiempo Promedio (min)": np.random.uniform(5, 15),
            "Tasa de Precisión (%)": np.random.randint(85, 99),
        }

# Función para aplicar el modelo de predicción
def predict_data(current_data, model_type="Linear Regression"):
    # Convertimos los datos actuales en un formato adecuado para la regresión
    x = np.array(list(range(len(current_data)))).reshape(-1, 1)
    y = np.array(list(current_data.values())).reshape(-1, 1)

    # Elegir el modelo de predicción
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100)
    else:
        model = LinearRegression()

    model.fit(x, y)

    # Predicción para el siguiente valor
    future_value = model.predict(np.array([[len(current_data)]]))
    
    # Devolver solo el valor escalar predicho
    return future_value[0]  # Ya es un valor escalar, no necesita más indexación

# Layout principal
st.title("💼 Tablero Multiusuario Dinámico con Predicciones")
st.markdown("Este tablero muestra datos personalizados y proyecciones para diferentes roles, todo actualizado en tiempo real.")

# Sidebar con opciones para el usuario
st.sidebar.title("⚙️ Configuración")
selected_role = st.sidebar.selectbox("Selecciona tu Rol", ["Director", "Analista"])

# Agregar un selector para elegir qué parámetro mostrar en el sidebar
selected_parameter = st.sidebar.selectbox(
    "Selecciona un parámetro para mostrar",
    ["Producción Diaria", "Ingresos Totales", "Rentabilidad (%)", "Eficiencia Operativa (%)"] if selected_role == "Director" else ["Análisis Completados", "Errores Detectados", "Tiempo Promedio (min)", "Tasa de Precisión (%)"]
)

# Agregar un selector para elegir el modelo de predicción
model_type = st.sidebar.selectbox("Selecciona el Modelo de Predicción", ["Linear Regression", "Random Forest"])

# Agregar la sección de ayuda en el sidebar
st.sidebar.markdown("### 🛠️ Ayuda")
st.sidebar.markdown("""
    Este tablero muestra indicadores clave para diferentes roles, como Director y Analista, con proyecciones de los próximos valores.
    
    ### Funcionalidad:
    - **Director:** Puede visualizar indicadores de producción, ingresos, rentabilidad y eficiencia operativa.
    - **Analista:** Puede ver estadísticas relacionadas con análisis completados, errores detectados, tiempo promedio y tasa de precisión.
    
    ### ¿Cómo interactuar?
    - Selecciona tu rol en la barra lateral para ver los indicadores correspondientes.
    - Elige un parámetro específico para visualizar y ver su valor actual.
    - Los indicadores se actualizan en tiempo real con predicciones basadas en modelos de regresión.
    
    ### Visualizaciones:
    - Los datos dinámicos son mostrados en un gráfico de barras, mostrando los indicadores actuales y las proyecciones futuras.

    # Agregar tu nombre en la parte inferior del sidebar
    st.sidebar.markdown("Creado por: Javier Horacio Pérez Ricárdez", unsafe_allow_html=True)
    
""")

# Contenedores para mantener actualizaciones en el mismo lugar
kpi_container = st.empty()
dataframe_container = st.empty()
chart_container = st.empty()

# Sidebar para mostrar KPIs dinámicos en una sola fila
sidebar_kpi_container = st.sidebar.empty()

while True:
    # Generar y actualizar datos dinámicos
    user_data = generate_user_data(selected_role)
    predicted_value = predict_data(user_data, model_type)

    # Actualizar los KPIs en el sidebar dinámicamente
    with sidebar_kpi_container.container():
        st.subheader(f"📊 KPIs para {selected_role}")
        sidebar_kpi_cols = st.columns(len(user_data))
        for i, (key, value) in enumerate(user_data.items()):
            sidebar_kpi_cols[i].metric(label=key, value=f"{value:.2f}" if isinstance(value, float) else value)

        # Mostrar solo el parámetro seleccionado en el sidebar
        st.subheader(f"📊 {selected_parameter} para {selected_role}")
        st.write(f"El valor actual de {selected_parameter} es: {user_data[selected_parameter]:.2f}")

    # Actualizar el DataFrame
    data_df = pd.DataFrame([user_data])
    data_df['Predicción del siguiente valor'] = predicted_value
    with dataframe_container:
        st.subheader("📋 Datos Dinámicos y Predicción")
        st.dataframe(data_df, use_container_width=True)

    # Actualizar el gráfico dinámico
    with chart_container:
        st.subheader("📈 Evolución de Indicadores y Proyección Futura")
        fig = px.bar(
            x=list(user_data.keys()) + ["Próximo valor"],
            y=list(user_data.values()) + [predicted_value],
            labels={"x": "Indicador", "y": "Valor"},
            title=f"Indicadores y Proyección para {selected_role}",
        )
        
        # Añadir un key único usando el rol y el timestamp actual
        st.plotly_chart(fig, use_container_width=True, key=f"bar_chart_{selected_role}_{time.time()}")

    time.sleep(2)




