import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from auto_detection import all_results

def generate_anomaly_report():
    st.title("Resultados de Modelos de Detección de Anomalías")

    try:
        # Crear un DataFrame a partir de los resultados
        df_results = pd.DataFrame(all_results)

        if df_results.empty:
            st.warning("No hay resultados para mostrar. Analiza al menos un modelo primero.")
            return

        # Mostrar gráfico comparativo de métricas
        st.subheader("Comparación de Métricas")
        plot_metrics(df_results)

        # Mostrar sección de detección de anomalías
        st.subheader("Detección de Anomalías")
        show_anomalies(df_results)

    except Exception as e:
        st.error(f"Error al cargar los resultados: {str(e)}")

def plot_metrics(df):
    # Crear un gráfico comparativo de métricas para cada modelo
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(x=df.index, y='accuracy', data=df, ax=ax, label='Accuracy')
    sns.barplot(x=df.index, y='precision', data=df, ax=ax, label='Precision')
    sns.barplot(x=df.index, y='recall', data=df, ax=ax, label='Recall')
    sns.barplot(x=df.index, y='f1_score', data=df, ax=ax, label='F1 Score')
    sns.barplot(x=df.index, y='roc_auc', data=df, ax=ax, label='Roc Auc')

    ax.set(title='Comparación de Métricas de Modelos',
           xlabel='Modelo',
           ylabel='Porcentaje')

    ax.legend()
    st.pyplot(fig)

def show_anomalies(df):
    # Mostrar una tabla con los resultados detallados de detección de anomalías
    st.dataframe(df)
